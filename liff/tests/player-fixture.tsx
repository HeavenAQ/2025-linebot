import React from 'react'
import { createRoot } from 'react-dom/client'
import VideoComparison from '../src/components/VideoComparison'
import { PlaybackResponseSchema } from '../src/schemas/userData.schema'

const marker = (id: string, label: string, start: number, end: number, duration: number) => ({
  id,
  label,
  timestamp_seconds: end,
  start_seconds: start,
  end_seconds: end,
  normalized_frame: end * 30,
  normalized_position: end / duration
})
const media = (name: string, duration: number) => ({
  signed_url: '/' + name + '.mp4',
  duration_seconds: duration,
  width: 320,
  height: 240,
  fps: 30
})
const playback = PlaybackResponseSchema.parse({
  analysis_id: 'synthetic-test',
  handedness: 'right',
  student_video: media('feedback', 7),
  skeleton_overlay_video: media('student', 4),
  expert: {
    expert_id: 'test',
    display_name: 'test',
    video: media('expert', 6),
    motion_start_seconds: 0,
    motion_end_seconds: 6,
    timeline: [
      marker('body_rotation', '身體旋轉', 1, 3, 6),
      marker('follow_through', '隨揮', 4, 5.9, 6)
    ]
  },
  timeline: [
    marker('body_rotation', '身體旋轉', 0.5, 1.4, 4),
    marker('follow_through', '隨揮', 2, 3.9, 4)
  ],
  coaching_cues: [
    {
      title: 'test',
      feedback: 'test',
      normalized_frame: 42,
      normalized_position: 0.35,
      student_timestamp_seconds: 1.4,
      pause_duration_seconds: 3,
      joint_ids: []
    }
  ],
  overall_feedback: '',
  grade: { total_grade: 60, grading_details: [] }
})
createRoot(document.getElementById('root')!).render(<VideoComparison playback={playback} />)
