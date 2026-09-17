'use strict';

(function () {
  // ---------------------------------------------------------------------------
  // Constants
  // ---------------------------------------------------------------------------

  var SESSION_CODE = 'gptReview.code';
  var SESSION_ROLE = 'gptReview.role';

  var SKILL_LABELS = { serve: '發球', smash: '殺球' };

  var CUE_OPTIONS = [
    { value: 'correct', label: '正確' },
    { value: 'partial', label: '部分正確' },
    { value: 'incorrect', label: '錯誤' }
  ];

  var SCORE_OPTIONS = [
    { value: 0, label: '錯誤或可能有害' },
    { value: 1, label: '部分正確' },
    { value: 2, label: '正確但不完整' },
    { value: 3, label: '正確且有用' }
  ];

  var STEP1_INSTRUCTION = '請依影片判斷下列檢核點是否需要改進（勾選＝需要改進；未勾選＝動作正確）。完成後才會顯示 GPT 的回饋，且步驟一送出後無法修改。';

  var EXPORTS = [
    { name: 'items.csv', label: '項目清單', desc: '每個影片項目一列' },
    { name: 'criteria.csv', label: '檢核點判斷', desc: '每位專家 × 項目 × 檢核點一列（步驟一）' },
    { name: 'cues.csv', label: 'GPT 建議評分', desc: '每位專家 × 項目 × GPT 建議一列' },
    { name: 'overall.csv', label: '整體評分與意見', desc: '每位專家 × 項目一列' }
  ];

  var MAX_COMMENT = 2000;

  var app = document.getElementById('app');
  var topnav = document.getElementById('topnav');
  var toastEl = document.getElementById('toast');

  var state = {
    items: null,   // cached GET /api/items list, in this expert's order
    dirty: false,  // unsaved step 2 changes
    renderToken: 0 // guards against late responses rendering over a newer page
  };

  // ---------------------------------------------------------------------------
  // DOM helpers (no innerHTML with data: everything is text nodes)
  // ---------------------------------------------------------------------------

  function h(tag, attrs) {
    var el = document.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (key) {
        var value = attrs[key];
        if (value === null || value === undefined || value === false) return;
        if (key === 'class') el.className = value;
        else if (key === 'text') el.textContent = value;
        else if (key.slice(0, 2) === 'on') el.addEventListener(key.slice(2), value);
        else if (value === true) el.setAttribute(key, '');
        else el.setAttribute(key, String(value));
      });
    }
    for (var i = 2; i < arguments.length; i++) append(el, arguments[i]);
    return el;
  }

  function append(parent, child) {
    if (child === null || child === undefined || child === false) return;
    if (Array.isArray(child)) {
      child.forEach(function (c) { append(parent, c); });
    } else if (typeof child === 'string' || typeof child === 'number') {
      parent.appendChild(document.createTextNode(String(child)));
    } else {
      parent.appendChild(child);
    }
  }

  function clear(el) {
    while (el.firstChild) el.removeChild(el.firstChild);
  }

  var uid = 0;
  function nextId(prefix) {
    uid += 1;
    return prefix + '-' + uid;
  }

  function mount() {
    clear(app);
    for (var i = 0; i < arguments.length; i++) append(app, arguments[i]);
    var heading = app.querySelector('h1');
    if (heading) {
      heading.setAttribute('tabindex', '-1');
      heading.focus({ preventScroll: true });
    }
    window.scrollTo(0, 0);
  }

  function showLoading() {
    clear(app);
    app.appendChild(h('p', { class: 'loading', text: '載入中…' }));
  }

  var toastTimer = null;
  function toast(message) {
    toastEl.textContent = message;
    toastEl.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { toastEl.hidden = true; }, 4000);
  }

  function alertBox(message, kind) {
    return h('div', { class: 'alert alert-' + (kind || 'error'), role: 'alert', text: message });
  }

  function setAlert(container, message, kind) {
    clear(container);
    if (message) container.appendChild(alertBox(message, kind));
  }

  function skillLabel(skill) {
    return SKILL_LABELS[skill] || '其他';
  }

  // ---------------------------------------------------------------------------
  // Session and API
  // ---------------------------------------------------------------------------

  function storageGet(key) {
    try { return sessionStorage.getItem(key); } catch (e) { return null; }
  }
  function storageSet(key, value) {
    try { sessionStorage.setItem(key, value); } catch (e) { /* private mode */ }
  }
  function storageClear() {
    try {
      sessionStorage.removeItem(SESSION_CODE);
      sessionStorage.removeItem(SESSION_ROLE);
    } catch (e) { /* ignore */ }
  }

  var memorySession = { code: null, role: null };

  function session() {
    return {
      code: storageGet(SESSION_CODE) || memorySession.code,
      role: storageGet(SESSION_ROLE) || memorySession.role
    };
  }

  function ApiError(status, code, retryAfter) {
    this.name = 'ApiError';
    this.status = status;
    this.code = code || '';
    this.retryAfter = retryAfter;
    this.message = 'API error ' + status;
  }
  ApiError.prototype = Object.create(Error.prototype);

  function logout(message) {
    storageClear();
    memorySession = { code: null, role: null };
    state.items = null;
    state.dirty = false;
    renderNav();
    renderLogin(message);
    if (location.hash && location.hash !== '#/') {
      history.replaceState(null, '', location.pathname);
      currentHash = '';
    }
  }

  function api(method, path, body, options) {
    options = options || {};
    var headers = { 'Accept': 'application/json' };
    var code = options.code || session().code;
    if (code) headers['X-Review-Code'] = code;
    var init = { method: method, headers: headers, cache: 'no-store', credentials: 'same-origin' };
    if (body !== undefined) {
      headers['Content-Type'] = 'application/json';
      init.body = JSON.stringify(body);
    }
    return fetch(path, init).then(function (res) {
      if (res.ok) return options.raw ? res : res.json();
      return res.json().catch(function () { return {}; }).then(function (data) {
        var err = new ApiError(res.status, data && data.error, res.headers.get('Retry-After'));
        if (res.status === 401 && !options.isLogin) {
          logout('登入已失效，請重新輸入審查代碼。');
          err.handled = true;
        }
        throw err;
      });
    }, function () {
      throw new ApiError(0, 'network');
    });
  }

  function errorMessage(err) {
    if (!(err instanceof ApiError)) return '發生未預期的錯誤，請重新整理頁面後再試。';
    switch (err.status) {
      case 0:
        return '無法連線到伺服器，請檢查網路連線後再試一次。';
      case 401:
        return '審查代碼不正確，請確認後再試一次。';
      case 403:
        return '您的帳號沒有使用這個功能的權限。';
      case 404:
        return '找不到這個項目，請回到列表重新選擇。';
      case 409:
        if (err.code === 'step1_locked') return '步驟一已經送出過，無法再修改。';
        if (err.code === 'step1_required') return '請先完成並送出步驟一。';
        return '資料狀態已變更，請重新整理頁面。';
      case 413:
        return '送出的內容太長，請縮短意見後再試。';
      case 429: {
        var seconds = parseInt(err.retryAfter, 10);
        var minutes = isNaN(seconds) ? 10 : Math.max(1, Math.ceil(seconds / 60));
        return '錯誤嘗試次數過多，請於約 ' + minutes + ' 分鐘後再試。';
      }
      case 400:
        return '送出的資料不完整或格式不正確，請檢查後再試。';
      default:
        return '伺服器暫時發生問題，請稍後再試。';
    }
  }

  function loadItems(force) {
    if (state.items && !force) return Promise.resolve(state.items);
    return api('GET', '/api/items').then(function (data) {
      state.items = (data && data.items) || [];
      return state.items;
    });
  }

  function updateCachedItem(itemId, patch) {
    if (!state.items) return;
    state.items.forEach(function (it) {
      if (it.item_id === itemId) Object.keys(patch).forEach(function (k) { it[k] = patch[k]; });
    });
  }

  function nextIncomplete(items, afterId) {
    var start = 0;
    if (afterId) {
      for (var i = 0; i < items.length; i++) {
        if (items[i].item_id === afterId) { start = i + 1; break; }
      }
    }
    for (var j = 0; j < items.length; j++) {
      var candidate = items[(start + j) % items.length];
      if (!candidate.completed && candidate.item_id !== afterId) return candidate;
    }
    return null;
  }

  // ---------------------------------------------------------------------------
  // Routing with an unsaved-changes guard
  // ---------------------------------------------------------------------------

  var currentHash = location.hash;
  var dialog = document.getElementById('dialog');

  function confirmDialog(title, message, confirmText, cancelText) {
    return new Promise(function (resolve) {
      document.getElementById('dialog-title').textContent = title;
      document.getElementById('dialog-message').textContent = message;
      var ok = document.getElementById('dialog-confirm');
      var cancel = document.getElementById('dialog-cancel');
      ok.textContent = confirmText;
      cancel.textContent = cancelText;
      var done = false;
      function finish(result) {
        if (done) return;
        done = true;
        ok.removeEventListener('click', onOk);
        cancel.removeEventListener('click', onCancel);
        dialog.removeEventListener('cancel', onCancel);
        if (dialog.open) dialog.close();
        resolve(result);
      }
      function onOk() { finish(true); }
      function onCancel(e) { if (e) e.preventDefault(); finish(false); }
      ok.addEventListener('click', onOk);
      cancel.addEventListener('click', onCancel);
      dialog.addEventListener('cancel', onCancel);
      if (typeof dialog.showModal === 'function') {
        dialog.showModal();
        cancel.focus();
      } else {
        // Very old browsers without <dialog>: leaving is the safe default.
        finish(true);
      }
    });
  }

  function confirmLeave() {
    if (!state.dirty) return Promise.resolve(true);
    return confirmDialog(
      '尚未儲存的評分',
      '步驟二的評分尚未送出，離開此頁面將會遺失這些變更。確定要離開嗎？',
      '離開，不儲存',
      '留在此頁'
    ).then(function (leave) {
      if (leave) state.dirty = false;
      return leave;
    });
  }

  function navigate(hash) {
    confirmLeave().then(function (leave) {
      if (!leave) return;
      if (location.hash === hash) route();
      else location.hash = hash;
    });
  }

  window.addEventListener('hashchange', function () {
    if (state.dirty) {
      var target = location.hash;
      history.replaceState(null, '', currentHash || location.pathname);
      confirmLeave().then(function (leave) {
        if (leave) location.hash = target;
      });
      return;
    }
    route();
  });

  window.addEventListener('beforeunload', function (e) {
    if (state.dirty) {
      e.preventDefault();
      e.returnValue = '';
    }
  });

  function route() {
    currentHash = location.hash;
    state.dirty = false;
    state.renderToken += 1;
    var s = session();
    renderNav();
    if (!s.code) {
      renderLogin();
      return;
    }
    var hash = location.hash.replace(/^#/, '');
    var itemMatch = /^\/item\/([A-Za-z0-9_.-]+)$/.exec(hash);
    if (s.role === 'admin') {
      renderAdmin();
    } else if (itemMatch) {
      renderItem(itemMatch[1]);
    } else {
      renderList();
    }
  }

  function renderNav() {
    clear(topnav);
    var s = session();
    if (!s.code) return;
    if (s.role !== 'admin') {
      topnav.appendChild(h('button', {
        type: 'button', class: 'btn btn-link', text: '項目列表',
        onclick: function () { navigate('#/'); }
      }));
    }
    topnav.appendChild(h('button', {
      type: 'button', class: 'btn btn-link', text: '登出',
      onclick: function () {
        confirmLeave().then(function (leave) { if (leave) logout(); });
      }
    }));
  }

  // ---------------------------------------------------------------------------
  // Login
  // ---------------------------------------------------------------------------

  function renderLogin(message) {
    var errorBox = h('div', { class: 'form-alert', 'aria-live': 'assertive' });
    var input = h('input', {
      id: 'login-code', name: 'code', type: 'password', class: 'input',
      autocomplete: 'off', autocapitalize: 'off', spellcheck: 'false', required: true
    });
    var button = h('button', { type: 'submit', class: 'btn btn-primary btn-block', text: '登入' });
    var form = h('form', { class: 'card login-card', novalidate: true },
      h('h1', { class: 'page-title', text: '專家審查登入' }),
      h('p', { class: 'muted', text: '請輸入研究團隊提供給您的審查代碼。' }),
      errorBox,
      h('label', { for: 'login-code', class: 'field-label', text: '審查代碼' }),
      input,
      button
    );
    if (message) setAlert(errorBox, message, 'info');

    form.addEventListener('submit', function (e) {
      e.preventDefault();
      var code = input.value.trim();
      if (!code) {
        setAlert(errorBox, '請輸入審查代碼。');
        input.focus();
        return;
      }
      button.disabled = true;
      button.textContent = '登入中…';
      api('POST', '/api/login', { code: code }, { isLogin: true, code: code }).then(function (data) {
        storageSet(SESSION_CODE, code);
        storageSet(SESSION_ROLE, data.role);
        memorySession = { code: code, role: data.role };
        state.items = null;
        if (location.hash === '#/' || !location.hash) route();
        else location.hash = '#/';
      }).catch(function (err) {
        button.disabled = false;
        button.textContent = '登入';
        setAlert(errorBox, errorMessage(err));
        input.select();
        input.focus();
      });
    });

    mount(h('section', { class: 'narrow' }, form));
    input.focus();
  }

  // ---------------------------------------------------------------------------
  // Item list and progress
  // ---------------------------------------------------------------------------

  function statusChip(item) {
    if (item.completed) return h('span', { class: 'chip chip-done', text: '已完成' });
    if (item.step1_done) return h('span', { class: 'chip chip-partial', text: '步驟一完成' });
    return h('span', { class: 'chip chip-todo', text: '未開始' });
  }

  function progressBar(done, total, label) {
    var pct = total > 0 ? Math.round((done / total) * 100) : 0;
    var fill = h('div', { class: 'progress-fill' });
    fill.style.width = pct + '%';
    return h('div', {
      class: 'progress', role: 'progressbar', 'aria-label': label,
      'aria-valuemin': 0, 'aria-valuemax': total, 'aria-valuenow': done
    }, fill);
  }

  function renderList() {
    var token = state.renderToken;
    showLoading();
    loadItems(true).then(function (items) {
      if (token !== state.renderToken) return;
      var total = items.length;
      var done = items.filter(function (it) { return it.completed; }).length;
      var next = nextIncomplete(items, null);

      var header = h('section', { class: 'card progress-card' },
        h('h1', { class: 'page-title', text: '審查進度' }),
        h('p', { class: 'progress-text' },
          '已完成 ', h('strong', { text: String(done) }), ' / ' + total + ' 項'),
        progressBar(done, total, '審查進度'),
        total === 0
          ? h('p', { class: 'muted', text: '目前沒有需要審查的項目。' })
          : next
            ? h('button', {
              type: 'button', class: 'btn btn-primary btn-large', text: '繼續下一個',
              onclick: function () { navigate('#/item/' + next.item_id); }
            })
            : h('p', { class: 'all-done', text: '所有項目皆已完成，感謝您的協助！您仍可點選下方項目檢視或修改步驟二。' })
      );

      var list = h('ul', { class: 'item-list', 'aria-label': '審查項目' });
      items.forEach(function (it, index) {
        list.appendChild(h('li', null,
          h('a', { class: 'item-row', href: '#/item/' + it.item_id },
            h('span', { class: 'item-index', 'aria-hidden': 'true', text: String(index + 1) }),
            h('span', { class: 'item-code', text: it.display_code }),
            h('span', { class: 'item-skill', text: skillLabel(it.skill) }),
            statusChip(it)
          )
        ));
      });

      mount(h('div', { class: 'wide' },
        header,
        total > 0 ? h('section', { class: 'list-section' },
          h('h2', { class: 'section-title', text: '全部項目' }),
          list
        ) : null
      ));
    }).catch(function (err) {
      if (token !== state.renderToken || err.handled) return;
      renderFailure(err, function () { renderList(); });
    });
  }

  function renderFailure(err, retry) {
    mount(h('section', { class: 'narrow' },
      h('div', { class: 'card' },
        h('h1', { class: 'page-title', text: '無法載入' }),
        alertBox(errorMessage(err)),
        h('div', { class: 'actions' },
          retry ? h('button', { type: 'button', class: 'btn btn-primary', text: '重試', onclick: retry }) : null,
          h('button', { type: 'button', class: 'btn btn-secondary', text: '返回列表', onclick: function () { navigate('#/'); } })
        )
      )
    ));
  }

  // ---------------------------------------------------------------------------
  // Item page
  // ---------------------------------------------------------------------------

  function renderItem(itemId) {
    var token = state.renderToken;
    showLoading();
    Promise.all([
      api('GET', '/api/items/' + encodeURIComponent(itemId)),
      loadItems(false).catch(function () { return []; })
    ]).then(function (results) {
      if (token !== state.renderToken) return;
      var item = results[0];
      if (item.step1_done) renderStep2(item);
      else renderStep1(item);
    }).catch(function (err) {
      if (token !== state.renderToken || err.handled) return;
      renderFailure(err, err.status === 404 ? null : function () { renderItem(itemId); });
    });
  }

  function itemHeader(item, stepLabel) {
    var position = '';
    if (state.items) {
      for (var i = 0; i < state.items.length; i++) {
        if (state.items[i].item_id === item.item_id) {
          position = '第 ' + (i + 1) + ' / ' + state.items.length + ' 項';
          break;
        }
      }
    }
    return h('div', { class: 'item-header' },
      h('button', {
        type: 'button', class: 'btn btn-link back-link', text: '← 返回列表',
        onclick: function () { navigate('#/'); }
      }),
      h('div', { class: 'item-title-row' },
        h('h1', { class: 'page-title' },
          h('span', { class: 'skill-badge skill-' + (item.skill === 'smash' ? 'smash' : 'serve'), text: skillLabel(item.skill) }),
          ' ',
          h('span', { text: item.display_code })
        ),
        position ? h('span', { class: 'muted small', text: position }) : null
      ),
      h('ol', { class: 'steps', 'aria-label': '審查步驟' },
        h('li', { class: stepLabel === 1 ? 'step current' : 'step done', 'aria-current': stepLabel === 1 ? 'step' : null, text: '步驟一：影片判斷' }),
        h('li', { class: stepLabel === 2 ? 'step current' : 'step', 'aria-current': stepLabel === 2 ? 'step' : null, text: '步驟二：評估 GPT 回饋' })
      )
    );
  }

  // videoBlock renders a looping muted player with playback-rate buttons.
  // refreshUrl() re-fetches a fresh signed URL when the old one has expired.
  function videoBlock(url, label, refreshUrl) {
    var wrapper = h('figure', { class: 'video-block' });
    var status = h('div', { class: 'video-status', 'aria-live': 'polite' });
    var video = h('video', {
      class: 'video', controls: true, playsinline: true, 'webkit-playsinline': true,
      muted: true, loop: true, preload: 'auto', 'aria-label': label
    });
    video.muted = true;
    video.defaultMuted = true;
    var retried = false;

    function setSource(src) {
      clear(status);
      if (!src) {
        status.appendChild(alertBox('影片暫時無法載入，請稍後重新整理頁面。'));
        video.hidden = true;
        return;
      }
      video.hidden = false;
      video.src = src;
      video.load();
    }

    video.addEventListener('error', function () {
      if (!retried && refreshUrl) {
        retried = true;
        refreshUrl().then(function (fresh) { setSource(fresh); }).catch(function () { setSource(''); });
        return;
      }
      setSource('');
    });

    var rates = [0.5, 1];
    var rateButtons = rates.map(function (rate) {
      return h('button', {
        type: 'button', class: 'btn btn-chip', 'aria-pressed': rate === 1 ? 'true' : 'false',
        text: rate + '×',
        onclick: function () {
          video.playbackRate = rate;
          rateButtons.forEach(function (b, i) { b.setAttribute('aria-pressed', rates[i] === rate ? 'true' : 'false'); });
        }
      });
    });
    video.addEventListener('loadedmetadata', function () {
      var active = rateButtons.filter(function (b) { return b.getAttribute('aria-pressed') === 'true'; })[0];
      if (active) video.playbackRate = rates[rateButtons.indexOf(active)];
    });

    append(wrapper, [
      video,
      status,
      h('figcaption', { class: 'video-controls' },
        h('span', { class: 'muted small', text: '播放速度' }),
        h('span', { class: 'rate-group', role: 'group', 'aria-label': '播放速度' }, rateButtons)
      )
    ]);
    setSource(url);
    return wrapper;
  }

  function refreshItemUrl(itemId, key) {
    return function () {
      return api('GET', '/api/items/' + encodeURIComponent(itemId)).then(function (fresh) { return fresh[key] || ''; });
    };
  }

  // --- Step 1 ---------------------------------------------------------------

  function renderStep1(item) {
    var openedAt = performance.now();
    var boxes = [];

    var checklist = h('fieldset', { class: 'checklist' },
      h('legend', { class: 'section-title', text: '需要改進的檢核點' }),
      item.criteria.map(function (c) {
        var id = nextId('crit');
        var box = h('input', { type: 'checkbox', id: id, value: c.id, class: 'check-input' });
        boxes.push(box);
        return h('label', { for: id, class: 'check-row' },
          box,
          h('span', { class: 'check-mark', 'aria-hidden': 'true' }),
          h('span', { class: 'check-text', text: c.name_zh })
        );
      })
    );

    var formAlert = h('div', { class: 'form-alert', 'aria-live': 'assertive' });
    var submit = h('button', { type: 'button', class: 'btn btn-primary btn-large btn-block', text: '送出步驟一並顯示 GPT 回饋' });
    var confirmPanel = h('div', { class: 'confirm-panel', hidden: true, role: 'region', 'aria-label': '確認送出步驟一' });

    function selections() {
      var out = {};
      boxes.forEach(function (b) { out[b.value] = b.checked; });
      return out;
    }

    function closeConfirm() {
      confirmPanel.hidden = true;
      submit.hidden = false;
      boxes.forEach(function (b) { b.disabled = false; });
      submit.focus();
    }

    function openConfirm() {
      setAlert(formAlert, '');
      var chosen = item.criteria.filter(function (c) { return selections()[c.id]; });
      clear(confirmPanel);
      var yes = h('button', { type: 'button', class: 'btn btn-primary', text: '確定送出' });
      var no = h('button', { type: 'button', class: 'btn btn-secondary', text: '返回修改' });
      append(confirmPanel, [
        h('p', { class: 'confirm-title', text: '確定要送出步驟一嗎？送出後將無法修改。' }),
        chosen.length
          ? h('div', null,
            h('p', { class: 'muted small', text: '您勾選為「需要改進」的檢核點（' + chosen.length + ' 項）：' }),
            h('ul', { class: 'plain-list' }, chosen.map(function (c) { return h('li', { text: c.name_zh }); })))
          : h('p', { class: 'muted small', text: '您沒有勾選任何檢核點，表示全部檢核點的動作皆正確。' }),
        h('div', { class: 'actions' }, no, yes)
      ]);
      no.addEventListener('click', closeConfirm);
      yes.addEventListener('click', function () {
        yes.disabled = true;
        no.disabled = true;
        yes.textContent = '送出中…';
        var seconds = Math.round((performance.now() - openedAt) / 100) / 10;
        api('PUT', '/api/items/' + encodeURIComponent(item.item_id) + '/step1', {
          needs_improvement: selections(),
          seconds: seconds
        }).then(function (updated) {
          updateCachedItem(item.item_id, { step1_done: true });
          renderStep2(updated);
        }).catch(function (err) {
          if (err.handled) return;
          if (err.status === 409) {
            toast(errorMessage(err));
            renderItem(item.item_id);
            return;
          }
          yes.disabled = false;
          no.disabled = false;
          yes.textContent = '確定送出';
          setAlert(formAlert, errorMessage(err));
        });
      });
      boxes.forEach(function (b) { b.disabled = true; });
      submit.hidden = true;
      confirmPanel.hidden = false;
      yes.focus();
    }

    submit.addEventListener('click', openConfirm);

    mount(h('div', { class: 'item-page' },
      itemHeader(item, 1),
      h('section', { class: 'card' },
        h('h2', { class: 'section-title', text: '學員動作影片' }),
        videoBlock(item.detected_overlay_url, '學員動作骨架影片', refreshItemUrl(item.item_id, 'detected_overlay_url')),
        h('p', { class: 'instruction', text: STEP1_INSTRUCTION }),
        checklist,
        formAlert,
        submit,
        confirmPanel
      )
    ));
  }

  // --- Step 2 ---------------------------------------------------------------

  function radioGroup(name, options, selected, onChange, extraClass) {
    return options.map(function (opt) {
      var id = nextId(name);
      var input = h('input', {
        type: 'radio', id: id, name: name, value: String(opt.value), class: 'choice-input',
        checked: selected !== null && selected !== undefined && String(selected) === String(opt.value)
      });
      input.addEventListener('change', onChange);
      return h('label', { for: id, class: 'choice ' + (extraClass || '') + ' choice-' + opt.value },
        input,
        h('span', { class: 'choice-body' }, opt.render ? opt.render() : opt.label)
      );
    });
  }

  function renderStep2(item) {
    var openedAt = performance.now();
    var rating = item.rating || {};
    var cues = item.gpt_cues || [];
    var criteriaNames = {};
    item.criteria.forEach(function (c) { criteriaNames[c.id] = c.name_zh; });

    function markDirty() {
      state.dirty = true;
      setAlert(formAlert, '');
    }

    // Read-only summary of step 1.
    var needs = rating.needs_improvement || {};
    var summary = h('section', { class: 'card summary-card' },
      h('h2', { class: 'section-title', text: '您在步驟一的判斷（已鎖定）' }),
      h('ul', { class: 'summary-list' }, item.criteria.map(function (c) {
        var bad = !!needs[c.id];
        return h('li', { class: 'summary-row' },
          h('span', { text: c.name_zh }),
          h('span', { class: bad ? 'chip chip-warn' : 'chip chip-ok', text: bad ? '需要改進' : '動作正確' })
        );
      }))
    );

    var feedbackText = h('p', { class: 'gpt-text', text: item.gpt_overall_feedback || '（GPT 沒有提供整體回饋文字）' });

    var cueFieldsets = cues.map(function (cue) {
      var name = 'cue-' + cue.index;
      var existing = rating.cue_ratings ? rating.cue_ratings[String(cue.index)] : null;
      var checkpoint = criteriaNames[cue.criterion_id] || '';
      var legendId = nextId('cue-legend');
      return h('fieldset', { class: 'card cue-card', 'data-cue': String(cue.index), 'aria-labelledby': legendId },
        h('div', { id: legendId, class: 'cue-head' },
          h('span', { class: 'cue-number', text: '建議 ' + cue.index }),
          checkpoint ? h('span', { class: 'cue-checkpoint', text: '檢核點：' + checkpoint }) : null,
          cue.title && cue.title !== checkpoint ? h('h3', { class: 'cue-title', text: cue.title }) : null
        ),
        h('p', { class: 'gpt-text', text: cue.feedback }),
        h('p', { class: 'question', text: '這項建議是否正確？' }),
        h('div', { class: 'choice-row' }, radioGroup(name, CUE_OPTIONS, existing, markDirty, 'choice-cue'))
      );
    });

    var scoreOptions = SCORE_OPTIONS.map(function (opt) {
      return {
        value: opt.value,
        render: function () {
          return [h('span', { class: 'score-num', text: String(opt.value) }), h('span', { class: 'score-desc', text: opt.label })];
        }
      };
    });
    var scoreFieldset = h('fieldset', { class: 'card score-card', 'data-score': 'true' },
      h('legend', { class: 'section-title legend-in-card', text: 'GPT 回饋整體評分' }),
      h('p', { class: 'muted small', text: '請綜合 GPT 的整體回饋與各項建議給分。' }),
      h('div', { class: 'score-grid' }, radioGroup('overall', scoreOptions, rating.overall_score, markDirty, 'choice-score'))
    );

    var commentId = nextId('comment');
    var counter = h('span', { class: 'muted small counter', 'aria-live': 'polite' });
    var comment = h('textarea', { id: commentId, class: 'input textarea', rows: 4, maxlength: MAX_COMMENT });
    comment.value = rating.comment || '';
    function updateCounter() { counter.textContent = comment.value.length + ' / ' + MAX_COMMENT; }
    updateCounter();
    comment.addEventListener('input', function () { markDirty(); updateCounter(); });

    var formAlert = h('div', { class: 'form-alert', 'aria-live': 'assertive' });
    var save = h('button', { type: 'button', class: 'btn btn-primary btn-large btn-block', text: '送出並前往下一個' });

    save.addEventListener('click', function () {
      var cueRatings = {};
      var missing = null;
      cueFieldsets.forEach(function (fs) {
        var checked = fs.querySelector('input:checked');
        fs.classList.toggle('needs-answer', !checked);
        if (checked) cueRatings[fs.getAttribute('data-cue')] = checked.value;
        else if (!missing) missing = fs;
      });
      var scoreInput = scoreFieldset.querySelector('input:checked');
      scoreFieldset.classList.toggle('needs-answer', !scoreInput);
      if (!missing && !scoreInput) missing = scoreFieldset;
      if (missing) {
        setAlert(formAlert, '請完成所有 GPT 建議的評分與整體評分後再送出。');
        var first = missing.querySelector('input');
        missing.scrollIntoView({ behavior: 'smooth', block: 'center' });
        if (first) first.focus({ preventScroll: true });
        return;
      }
      if (comment.value.length > MAX_COMMENT) {
        setAlert(formAlert, '意見最多 ' + MAX_COMMENT + ' 字。');
        comment.focus();
        return;
      }
      save.disabled = true;
      save.textContent = '儲存中…';
      var seconds = Math.round((performance.now() - openedAt) / 100) / 10;
      api('PUT', '/api/items/' + encodeURIComponent(item.item_id) + '/step2', {
        cue_ratings: cueRatings,
        overall_score: parseInt(scoreInput.value, 10),
        comment: comment.value,
        seconds: seconds
      }).then(function () {
        state.dirty = false;
        updateCachedItem(item.item_id, { step1_done: true, completed: true });
        var next = state.items ? nextIncomplete(state.items, item.item_id) : null;
        if (next) {
          toast('已儲存 ' + item.display_code + '。');
          location.hash = '#/item/' + next.item_id;
        } else {
          toast('已儲存。所有項目皆已完成，感謝您的協助！');
          location.hash = '#/';
        }
      }).catch(function (err) {
        if (err.handled) return;
        save.disabled = false;
        save.textContent = '送出並前往下一個';
        setAlert(formAlert, errorMessage(err));
      });
    });

    mount(h('div', { class: 'item-page' },
      itemHeader(item, 2),
      rating.completed ? alertBox('此項目已送出過。您可以修改評分後再次送出。', 'info') : null,
      summary,
      h('section', { class: 'card' },
        h('h2', { class: 'section-title', text: 'GPT 回饋影片' }),
        videoBlock(item.feedback_video_url, 'GPT 回饋標記影片', refreshItemUrl(item.item_id, 'feedback_video_url'))
      ),
      h('section', { class: 'card' },
        h('h2', { class: 'section-title', text: 'GPT 整體回饋' }),
        feedbackText
      ),
      h('section', { class: 'cue-section' },
        h('h2', { class: 'section-title', text: 'GPT 提出的改進項目' }),
        cues.length ? cueFieldsets : h('p', { class: 'card empty-note', text: 'GPT 沒有提出具體改進項目' })
      ),
      scoreFieldset,
      h('section', { class: 'card' },
        h('label', { for: commentId, class: 'section-title', text: '意見（選填）' }),
        comment,
        counter
      ),
      formAlert,
      save
    ));
  }

  // ---------------------------------------------------------------------------
  // Admin
  // ---------------------------------------------------------------------------

  function renderAdmin() {
    var token = state.renderToken;
    showLoading();
    api('GET', '/api/admin/progress').then(function (data) {
      if (token !== state.renderToken) return;
      var experts = data.experts || [];
      var rows = experts.map(function (e) {
        return h('tr', null,
          h('th', { scope: 'row', text: e.expert_id }),
          h('td', { class: 'num', text: String(e.total) }),
          h('td', { class: 'num', text: String(e.step1_done) }),
          h('td', { class: 'num', text: String(e.completed) }),
          h('td', { class: 'bar-cell' }, progressBar(e.completed, e.total, e.expert_id + ' 完成進度'))
        );
      });
      var downloadAlert = h('div', { class: 'form-alert', 'aria-live': 'assertive' });

      var downloads = EXPORTS.map(function (x) {
        var button = h('button', { type: 'button', class: 'btn btn-secondary', text: '下載 ' + x.name });
        button.addEventListener('click', function () {
          button.disabled = true;
          setAlert(downloadAlert, '');
          api('GET', '/api/admin/export/' + x.name, undefined, { raw: true }).then(function (res) {
            return res.blob();
          }).then(function (blob) {
            var url = URL.createObjectURL(blob);
            var a = h('a', { href: url, download: x.name, class: 'visually-hidden' });
            document.body.appendChild(a);
            a.click();
            setTimeout(function () { URL.revokeObjectURL(url); a.remove(); }, 1000);
          }).catch(function (err) {
            if (!err.handled) setAlert(downloadAlert, errorMessage(err));
          }).then(function () { button.disabled = false; });
        });
        return h('li', { class: 'download-row' },
          h('div', null, h('strong', { text: x.label }), h('div', { class: 'muted small', text: x.desc })),
          button
        );
      });

      mount(h('div', { class: 'wide' },
        h('section', { class: 'card' },
          h('div', { class: 'item-title-row' },
            h('h1', { class: 'page-title', text: '管理：審查進度' }),
            h('button', { type: 'button', class: 'btn btn-secondary', text: '重新整理', onclick: function () { route(); } })
          ),
          data.batch_id ? h('p', { class: 'muted small', text: '批次：' + data.batch_id }) : null,
          h('div', { class: 'table-wrap' },
            h('table', { class: 'table' },
              h('thead', null, h('tr', null,
                h('th', { scope: 'col', text: '審查者' }),
                h('th', { scope: 'col', class: 'num', text: '項目總數' }),
                h('th', { scope: 'col', class: 'num', text: '步驟一完成' }),
                h('th', { scope: 'col', class: 'num', text: '已完成' }),
                h('th', { scope: 'col', text: '進度' })
              )),
              h('tbody', null, rows)
            )
          )
        ),
        h('section', { class: 'card' },
          h('h2', { class: 'section-title', text: '匯出資料（CSV）' }),
          downloadAlert,
          h('ul', { class: 'download-list' }, downloads)
        )
      ));
    }).catch(function (err) {
      if (token !== state.renderToken || err.handled) return;
      renderFailure(err, function () { route(); });
    });
  }

  // ---------------------------------------------------------------------------

  route();
})();
