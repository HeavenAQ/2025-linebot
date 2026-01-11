import { z } from "zod"

// Grading
export const GradingDetailSchema = z.object({
  description: z.string(),
  grade: z.number(),
})

export const GradingOutcomeSchema = z.object({
  grading_details: z.array(GradingDetailSchema),
  total_grade: z.number(),
})

// Work (single video entry)
export const WorkSchema = z.object({
  date: z.string(),
  thumbnail: z.url(),
  skeleton_video: z.url(),
  skeleton_comparison_video: z.url(),
  reflection: z.string(),
  preview_note: z.string(),
  ai_note: z.string(),
  grading_outcome: GradingOutcomeSchema,
})

// Portfolios
export const PortfoliosSchema = z.object({
  serve: z.record(z.string(), WorkSchema),
  smash: z.record(z.string(), WorkSchema),
  clear: z.record(z.string(), WorkSchema),
})

// Folder IDs
export const FolderIDsSchema = z.object({
  root: z.string(),
  serve: z.string(),
  smash: z.string(),
  clear: z.string(),
  thumbnail: z.string(),
})

// GPT Conversation IDs
export const GPTConversationIDsSchema = z.object({
  serve: z.string(),
  smash: z.string(),
  clear: z.string(),
})

// Root UserData

export const UserDataSchema = z.object({
  portfolio: PortfoliosSchema,
  folder_paths: FolderIDsSchema,
  gpt_conversation_ids: GPTConversationIDsSchema,
  name: z.string(),
  id: z.string(),
  handedness: z.number(),
})

// TypeScript Types (derived from Zod)
export type UserData = z.infer<typeof UserDataSchema>
export type Portfolios = z.infer<typeof PortfoliosSchema>
export type Work = z.infer<typeof WorkSchema>
export type GradingOutcome = z.infer<typeof GradingOutcomeSchema>
export type GradingDetail = z.infer<typeof GradingDetailSchema>
export type FolderIDs = z.infer<typeof FolderIDsSchema>
export type GPTConversationIDs = z.infer<typeof GPTConversationIDsSchema>

export default UserDataSchema
