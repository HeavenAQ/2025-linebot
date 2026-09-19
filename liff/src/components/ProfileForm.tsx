'use client'

import React, { useState } from 'react'

import { Button } from '@/components/ui/button'
import { InputField } from '@/components/ui/input'
import { SelectField } from '@/components/ui/select'
import { saveProfile } from '@/lib/api/profile'
import { NAME_HINT, NUMBER_HINT, normalizeName, normalizeNumber, validateProfile } from '@/lib/profileInput'
import type { ProfileErrors } from '@/lib/profileInput'
import type { UserData } from '@/types'

interface ProfileFormProps {
  /** The learner's stored details, when they already have some. */
  user?: UserData | null
  /** What the save button says, e.g. 完成登記 or 儲存變更. */
  submitLabel: string
  onSaved: (_user: UserData) => void
}

/**
 * The experiment registration form: the same three fields whether a student is
 * registering for the first time or correcting what they entered.
 */
export default function ProfileForm({ user, submitLabel, onSaved }: ProfileFormProps) {
  const [experimentNumber, setExperimentNumber] = useState(user?.experiment_number ?? '')
  const [realName, setRealName] = useState(user?.real_name ?? '')
  const [handedness, setHandedness] = useState<'left' | 'right'>(user?.handedness === 0 ? 'left' : 'right')
  const [errors, setErrors] = useState<ProfileErrors>({})
  const [saveError, setSaveError] = useState('')
  const [saving, setSaving] = useState(false)

  async function submit(event: React.FormEvent) {
    event.preventDefault()
    setSaveError('')
    const found = validateProfile({ experimentNumber, realName })
    setErrors(found)
    if (Object.keys(found).length > 0) return

    setSaving(true)
    try {
      const saved = await saveProfile({
        experiment_number: normalizeNumber(experimentNumber),
        real_name: normalizeName(realName),
        handedness
      })
      onSaved(saved)
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : '無法儲存資料，請稍後再試。')
    } finally {
      setSaving(false)
    }
  }

  return (
    <form onSubmit={submit} className="space-y-4" noValidate>
      <InputField
        label="實驗編號"
        value={experimentNumber}
        onChange={event => setExperimentNumber(event.target.value)}
        placeholder="01"
        inputMode="text"
        autoComplete="off"
        maxLength={10}
        hint={NUMBER_HINT}
        error={errors.experimentNumber}
      />
      <InputField
        label="姓名"
        value={realName}
        onChange={event => setRealName(event.target.value)}
        placeholder="王小明"
        autoComplete="name"
        maxLength={60}
        hint={NAME_HINT}
        error={errors.realName}
      />
      <SelectField
        label="慣用手"
        value={handedness}
        onChange={event => setHandedness(event.target.value as 'left' | 'right')}
      >
        <option value="right">右手</option>
        <option value="left">左手</option>
      </SelectField>
      <p className="text-xs text-muted-foreground">
        慣用手會用於動作分析與專家示範影片，之後可以在這裡修改。
      </p>

      {saveError && (
        <p role="alert" className="text-sm text-destructive">
          {saveError}
        </p>
      )}

      <Button type="submit" className="w-full" disabled={saving}>
        {saving ? '儲存中…' : submitLabel}
      </Button>
    </form>
  )
}
