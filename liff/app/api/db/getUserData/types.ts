export interface UserData {
  Portfolio: Portfolios
  FolderIDs: FolderIDs
  GPTThreadIDs: GPTThreadIDs
  Name: string
  Id: string
  Handedness: string
}

export interface FolderIDs {
  Root: string
  Serve: string
  Smash: string
  Clear: string
  Thumbnail: string
}

export interface Portfolios {
  Serve: Record<string, Work>
  Smash: Record<string, Work>
  Clear: Record<string, Work>
}

export interface GPTThreadIDs {
  Serve: string
  Smash: string
  Clear: string
}

export interface Work {
  DateTime: string
  Thumbnail: string
  SkeletonVideo: string
  SkeletonComparisonVideo: string
  Reflection: string
  PreviewNote: string
  AINote: string
  GradingOutcome: GradingOutcome
}

export interface GradingDetail {
  Description: string
  Grade: number
}

export interface GradingOutcome {
  GradingDetails: GradingDetail[]
  TotalGrade: number
}
