/**
 * Links from the bot into the review tab: the attempt a portfolio card names,
 * and the sub-tab a rich-menu card opens.
 *
 * A card names the attempt it shows (?tab=review&skill=…&date=…), but it stays
 * in the chat history for the rest of the semester — long enough to outlive the
 * work it points at. So the attempt is resolved against the portfolio that
 * actually loaded, and anything that no longer matches is dropped: a stale or
 * hand-edited link should leave the page on its usual default rather than open
 * an empty week.
 */

import type { Skill } from '@/lib/types'
import type { Portfolios } from '@/schemas/userData.schema'

export interface WorkFocus {
  skill: Skill
  date: string
}

/** Own keys only: a query string may name "constructor" as readily as a date. */
const owns = (record: object, key: string) => Object.prototype.hasOwnProperty.call(record, key)

/**
 * The attempt a review link points at, or null when there is no usable one.
 * `search` is a location search string, e.g. "?tab=review&skill=serve&date=…".
 */
export function resolveWorkFocus(search: string, portfolio: Portfolios): WorkFocus | null {
  const params = new URLSearchParams(search)
  const skill = params.get('skill')
  const date = params.get('date')
  if (!skill || !date) return null
  // The portfolio carries one map per skill, so it is also the list of skills
  // that exist — no separate table to keep in step with it.
  if (!owns(portfolio, skill)) return null
  const works = portfolio[skill as Skill]
  if (!works || !owns(works, date)) return null
  return { skill: skill as Skill, date }
}

/** The 每週回顧 sub-tabs, in the order they are shown. */
export const REVIEW_SECTIONS = ['reflection', 'preview'] as const

export type ReviewSection = (typeof REVIEW_SECTIONS)[number]

/**
 * The 每週回顧 sub-tab a link asks for, or null when it names none. A link can
 * send a learner straight to their preview note with ?tab=review&section=preview;
 * one naming anything else leaves the page on its own default, for the same
 * reason a stale attempt does.
 */
export function resolveReviewSection(search: string): ReviewSection | null {
  const section = new URLSearchParams(search).get('section')
  return REVIEW_SECTIONS.includes(section as ReviewSection) ? (section as ReviewSection) : null
}
