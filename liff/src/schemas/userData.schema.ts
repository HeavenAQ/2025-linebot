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

const OptionalMediaRefSchema = z.preprocess(value => value ?? {}, MediaRefSchema)

export const PhaseMarkerSchema = z.object({
  id: z.string(),
  label: z.string(),
  normalized_frame: z.number(),
  normalized_position: z.number(),
  timestamp_seconds: z.number()
})

export const AlignmentSampleSchema = z.object({
  normalized_position: z.number(),
  expert_seconds: z.number()
})

export const ExpertMatchSchema = z.object({
  expert_id: z.string(),
  display_name: z.string(),
  correction_distance: z.number().optional().default(0),
  video: MediaRefSchema,
  motion_start_seconds: z.number().optional().default(0),
  motion_end_seconds: z.number().optional().default(0),
  // The expert's own checkpoints, timestamped in the expert video. Absent on
  // analyses recorded before checkpoint alignment shipped.
  timeline: z.preprocess(value => value ?? [], z.array(PhaseMarkerSchema)),
  // The warped map between those checkpoints. Absent on analyses recorded
  // before segmental alignment shipped, and on any the service could not warp.
  alignment: z.preprocess(value => value ?? [], z.array(AlignmentSampleSchema))
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

const WorkHandednessSchema = z.preprocess(
  value => (value === '' || value == null ? 'right' : value),
  z.enum(['left', 'right'])
)

export const WorkSchema = z.object({
  date: z.string(),
  handedness: WorkHandednessSchema,
  thumbnail: z.string(),
  reflection: z.string(),
  ai_note: z.string(),
  grading_outcome: GradingOutcomeSchema,
  analysis_id: z.string().optional().default(''),
  student_video: MediaRefSchema.optional(),
  feedback_video: OptionalMediaRefSchema.optional(),
  skeleton_overlay_video: OptionalMediaRefSchema.optional(),
  expert: ExpertMatchSchema.optional(),
  timeline: z.preprocess(value => value ?? [], z.array(PhaseMarkerSchema)),
  coaching_cues: z.preprocess(value => value ?? [], z.array(CoachingCueSchema))
})

const EmptyPortfolio = z.record(z.string(), WorkSchema)
const NullablePortfolio = z.preprocess(value => value ?? {}, EmptyPortfolio)

export const PortfoliosSchema = z.object({
  serve: NullablePortfolio,
  smash: NullablePortfolio,
  clear: NullablePortfolio,
  lift: NullablePortfolio
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
  handedness: z.enum(['left', 'right']).optional().default('right'),
  student_video: MediaRefSchema,
  feedback_video: OptionalMediaRefSchema,
  skeleton_overlay_video: OptionalMediaRefSchema,
  expert: ExpertMatchSchema,
  timeline: z.array(PhaseMarkerSchema),
  coaching_cues: z.array(CoachingCueSchema),
  overall_feedback: z.string(),
  grade: GradingOutcomeSchema
})

export type Work = z.infer<typeof WorkSchema>
export type UserData = z.infer<typeof UserDataSchema>
export type Portfolios = z.infer<typeof PortfoliosSchema>
export type GradingDetail = z.infer<typeof GradingDetailSchema>
export type PhaseMarker = z.infer<typeof PhaseMarkerSchema>
export type AlignmentSample = z.infer<typeof AlignmentSampleSchema>
export type CoachingCue = z.infer<typeof CoachingCueSchema>
export type PlaybackResponse = z.infer<typeof PlaybackResponseSchema>

export default UserDataSchema
