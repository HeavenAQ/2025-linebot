import { z } from 'zod'

export const GradingDetailSchema = z.object({
  criterion_id: z.string().optional().default(''),
  description: z.string(),
  grade: z.number(),
  maximum: z.number().optional().default(20)
})

export const GradingOutcomeSchema = z.object({
  grading_details: z.array(GradingDetailSchema),
  total_grade: z.number(),
  score_status: z.string().optional().default('')
})

export const MediaRefSchema = z.object({
  object_path: z.string().optional().default(''),
  gcs_uri: z.string().optional().default(''),
  signed_url: z.string().optional().default(''),
  signed_url_expires_at_unix: z.number().optional().default(0),
  duration_seconds: z.number().optional().default(0),
  fps: z.number().optional().default(0),
  width: z.number().optional().default(0),
  height: z.number().optional().default(0)
})

export const ExpertMatchSchema = z.object({
  expert_id: z.string(),
  display_name: z.string(),
  euclidean_distance: z.number(),
  video: MediaRefSchema
})

export const PhaseMarkerSchema = z.object({
  id: z.string(),
  label: z.string(),
  normalized_frame: z.number(),
  normalized_position: z.number(),
  timestamp_seconds: z.number()
})

export const CoachingCueSchema = z.object({
  title: z.string(),
  feedback: z.string(),
  normalized_frame: z.number(),
  normalized_position: z.number(),
  student_timestamp_seconds: z.number(),
  pause_duration_seconds: z.number(),
  joint_ids: z.array(z.number())
})

export const WorkSchema = z.object({
  date: z.string(),
  thumbnail: z.string(),
  skeleton_video: z.string().optional().default(''),
  skeleton_comparison_video: z.string().optional().default(''),
  reflection: z.string(),
  preview_note: z.string(),
  ai_note: z.string(),
  grading_outcome: GradingOutcomeSchema,
  analysis_id: z.string().optional().default(''),
  student_video: MediaRefSchema.optional(),
  expert: ExpertMatchSchema.optional(),
  timeline: z.array(PhaseMarkerSchema).optional().default([]),
  coaching_cues: z.array(CoachingCueSchema).optional().default([])
})

const EmptyPortfolio = z.record(z.string(), WorkSchema)

export const PortfoliosSchema = z.object({
  serve: EmptyPortfolio,
  smash: EmptyPortfolio,
  clear: EmptyPortfolio,
  lift: EmptyPortfolio.optional().default({})
})

export const FolderIDsSchema = z.object({
  root: z.string(),
  serve: z.string(),
  smash: z.string(),
  clear: z.string(),
  lift: z.string().optional().default(''),
  thumbnail: z.string()
})

export const GPTConversationIDsSchema = z.object({
  serve: z.string(),
  smash: z.string(),
  clear: z.string(),
  lift: z.string().optional().default('')
})

export const UserDataSchema = z.object({
  portfolio: PortfoliosSchema,
  folder_paths: FolderIDsSchema,
  gpt_conversation_ids: GPTConversationIDsSchema,
  name: z.string(),
  id: z.string(),
  handedness: z.number()
})

export const PlaybackResponseSchema = z.object({
  analysis_id: z.string(),
  student_video: MediaRefSchema,
  expert: ExpertMatchSchema,
  timeline: z.array(PhaseMarkerSchema),
  coaching_cues: z.array(CoachingCueSchema),
  overall_feedback: z.string(),
  grade: GradingOutcomeSchema
})

export type UserData = z.infer<typeof UserDataSchema>
export type Portfolios = z.infer<typeof PortfoliosSchema>
export type Work = z.infer<typeof WorkSchema>
export type GradingOutcome = z.infer<typeof GradingOutcomeSchema>
export type GradingDetail = z.infer<typeof GradingDetailSchema>
export type FolderIDs = z.infer<typeof FolderIDsSchema>
export type GPTConversationIDs = z.infer<typeof GPTConversationIDsSchema>
export type MediaRef = z.infer<typeof MediaRefSchema>
export type ExpertMatch = z.infer<typeof ExpertMatchSchema>
export type PhaseMarker = z.infer<typeof PhaseMarkerSchema>
export type CoachingCue = z.infer<typeof CoachingCueSchema>
export type PlaybackResponse = z.infer<typeof PlaybackResponseSchema>

export default UserDataSchema
