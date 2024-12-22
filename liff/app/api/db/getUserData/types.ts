export interface UserData {
  portfolio: Portfolios
  folderIDs: FolderIDs
  gptThreadIDs: GPTThreadIDs
  name: string
  id: string
  handedness: string
}

export interface FolderIDs {
  root: string
  serve: string
  smash: string
  clear: string
  thumbnail: string
}

export interface Portfolios {
  serve: Record<string, Work>
  smash: Record<string, Work>
  clear: Record<string, Work>
}

export interface GPTThreadIDs {
  serve: string
  smash: string
  clear: string
}

export interface Work {
  dateTime: string
  thumbnail: string
  skeletonVideo: string
  skeletonComparisonVideo: string
  reflection: string
  previewNote: string
  aiNote: string
  gradingOutcome: GradingOutcome
}

export interface GradingDetail {
  description: string
  grade: number
}

export interface GradingOutcome {
  gradingDetails: GradingDetail[]
  totalGrade: number
}
