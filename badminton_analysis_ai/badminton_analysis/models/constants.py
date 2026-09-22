SMOOTHING_WINDOW_SIZE = 5
IMPACT_FRAME_SEARCH_WINDOW_BEFORE = 30
IMPACT_FRAME_SEARCH_WINDOW_AFTER = 30
ANALYSIS_WINDOW_PADDING_BEFORE = 30

# Frames that must follow serve contact for the follow-through and completion
# phases to be distinct after the clip is resampled.
SERVE_MINIMUM_FOLLOW_THROUGH = 8

# How far the hand must drop, as a share of its range over the clip, for a
# candidate contact frame to be a swing rather than a still moment.
SERVE_PEAK_DROP_RATIO = 0.6

# A serve's contact is the lowest point of its fast swing. The hand can go
# lower afterwards, as the racket is let down, so the lowest-hand search stays
# this close to the swing's acceleration peak.
SERVE_CONTACT_SEARCH_FRAMES = 8

# Below this confidence a wrist keypoint is bridged from its neighbours before
# the swing is searched for: a jump through a poorly seen wrist is not a swing.
SERVE_WRIST_CONFIDENCE_FLOOR = 0.8
