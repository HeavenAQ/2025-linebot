/**
 * The same rules the Go server applies in ParseExperimentRegistration, so a
 * student sees what is wrong while typing instead of after a round trip. The
 * server still decides: this only saves them the trip.
 */

export const EXPERIMENT_NUMBER_PATTERN = /^[A-Z]{0,4}[0-9]{1,6}$/
const NAME_ALLOWED = /^[\p{L}\p{M}'’\-·・ ]+$/u
const LETTER = /\p{L}/u

export const NUMBER_HINT = '編號可用數字，或英文字母加數字（例如 01、EG01）。'
export const NAME_HINT = '請填寫真實姓名，不可含數字、表情符號或其他說明。'

export interface ProfileInput {
  experimentNumber: string
  realName: string
}

export interface ProfileErrors {
  experimentNumber?: string
  realName?: string
}

/** normalizeNumber matches the server: full-width to ASCII, then upper case. */
export function normalizeNumber(value: string): string {
  return value
    .replace(/[！-～]/g, character => String.fromCharCode(character.charCodeAt(0) - 0xfee0))
    .trim()
    .toUpperCase()
}

/** normalizeName collapses the runs of spaces the server collapses. */
export function normalizeName(value: string): string {
  return value
    .replace(/[！-～]/g, character => String.fromCharCode(character.charCodeAt(0) - 0xfee0))
    .trim()
    .replace(/\s+/g, ' ')
}

export function validateProfile({ experimentNumber, realName }: ProfileInput): ProfileErrors {
  const errors: ProfileErrors = {}

  const number = normalizeNumber(experimentNumber)
  if (!number) errors.experimentNumber = '請輸入實驗編號。'
  else if (!EXPERIMENT_NUMBER_PATTERN.test(number)) errors.experimentNumber = NUMBER_HINT

  const name = normalizeName(realName)
  const letters = Array.from(name).filter(character => LETTER.test(character)).length
  if (!name) errors.realName = '請輸入姓名。'
  else if (Array.from(name).length < 2 || Array.from(name).length > 60) {
    errors.realName = '姓名長度需介於 2 至 60 個字。'
  } else if (!NAME_ALLOWED.test(name) || letters < 2) errors.realName = NAME_HINT
  else if (!LETTER.test(Array.from(name)[0])) errors.realName = NAME_HINT

  return errors
}

export function isValidProfile(input: ProfileInput): boolean {
  return Object.keys(validateProfile(input)).length === 0
}
