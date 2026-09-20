import cv2
import math


# ============================================================
# CONFIGURATION
# ============================================================

PIXELS_PER_ANT_METER = 100.0

# Contour filtering
MIN_AREA = 20
MAX_AREA = 2500

# Movement
MIN_MOVEMENT = 1.5
MAX_JUMP = 80

# Smoothing
SMOOTHING_ALPHA = 0.30

# Tracking
MAX_LOST_FRAMES = 15
MIN_CONFIRM_FRAMES = 4
MAX_CANDIDATE_JUMP = 50

# Stationary ant
STATIONARY_SEARCH_RADIUS = 35

# Camera motion
MAX_CAMERA_SHIFT = 30

# ORB
ORB_FEATURES = 500


# ============================================================
# CAMERA MOTION ESTIMATION
# ============================================================

def estimate_camera_motion(previous_gray, current_gray, orb):
    """
    Estimate global camera translation between two frames.

    Returns:
        dx, dy

    If estimation fails:
        0, 0
    """

    keypoints1, descriptors1 = orb.detectAndCompute(
        previous_gray,
        None
    )

    keypoints2, descriptors2 = orb.detectAndCompute(
        current_gray,
        None
    )

    if descriptors1 is None or descriptors2 is None:
        return 0.0, 0.0

    if len(keypoints1) < 5 or len(keypoints2) < 5:
        return 0.0, 0.0

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=True
    )

    matches = matcher.match(
        descriptors1,
        descriptors2
    )

    if len(matches) < 5:
        return 0.0, 0.0

    matches = sorted(
        matches,
        key=lambda m: m.distance
    )

    # Only use reasonably good matches
    matches = matches[:50]

    shifts_x = []
    shifts_y = []

    for match in matches:

        p1 = keypoints1[
            match.queryIdx
        ].pt

        p2 = keypoints2[
            match.trainIdx
        ].pt

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        # Ignore absurd camera movements
        if abs(dx) <= MAX_CAMERA_SHIFT and \
           abs(dy) <= MAX_CAMERA_SHIFT:

            shifts_x.append(dx)
            shifts_y.append(dy)

    if len(shifts_x) < 3:
        return 0.0, 0.0

    # Median is resistant to bad feature matches
    dx = float(
        sorted(shifts_x)[len(shifts_x) // 2]
    )

    dy = float(
        sorted(shifts_y)[len(shifts_y) // 2]
    )

    return dx, dy


# ============================================================
# APPLY CAMERA STABILIZATION
# ============================================================

def stabilize_frame(
    gray,
    dx,
    dy
):

    transform = cv2.getRotationMatrix2D(
        (gray.shape[1] / 2, gray.shape[0] / 2),
        0,
        1
    )

    transform[0, 2] -= dx
    transform[1, 2] -= dy

    stabilized = cv2.warpAffine(
        gray,
        transform,
        (
            gray.shape[1],
            gray.shape[0]
        ),
        borderMode=cv2.BORDER_REFLECT
    )

    return stabilized


# ============================================================
# CANDIDATE SCORING
# ============================================================

def score_candidate(
    candidate,
    previous_position,
    frame_width,
    frame_height
):

    score = 0.0

    x = candidate["x"]
    y = candidate["y"]
    area = candidate["area"]

    # --------------------------------------------------------
    # 1. Area score
    # --------------------------------------------------------

    # Ants are expected to be relatively small.
    ideal_area = 100

    area_difference = abs(
        area - ideal_area
    )

    area_score = max(
        0,
        1 - (
            area_difference /
            500
        )
    )

    score += area_score * 25

    # --------------------------------------------------------
    # 2. Shape score
    # --------------------------------------------------------

    aspect_ratio = candidate["aspect_ratio"]

    if 1.0 <= aspect_ratio <= 3.5:

        score += 20

    elif aspect_ratio <= 5:

        score += 10

    # --------------------------------------------------------
    # 3. Distance from previous ant
    # --------------------------------------------------------

    if previous_position is not None:

        distance = math.dist(
            previous_position,
            (x, y)
        )

        if distance <= 20:

            score += 40

        elif distance <= 40:

            score += 30

        elif distance <= 60:

            score += 15

        else:

            score -= 30

    # --------------------------------------------------------
    # 4. Avoid frame edges
    # --------------------------------------------------------

    margin = 10

    if (
        margin < x < frame_width - margin
        and
        margin < y < frame_height - margin
    ):

        score += 5

    # --------------------------------------------------------
    # 5. Compact object bonus
    # --------------------------------------------------------

    width = candidate["w"]
    height = candidate["h"]

    if width <= 100 and height <= 100:

        score += 10

    return score


# ============================================================
# MAIN VIDEO ANALYSIS
# ============================================================

def analyze_video(video_path: str):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open video")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    # ============================================================
    # ORB camera motion detector
    # ============================================================

    orb = cv2.ORB_create(
        nfeatures=ORB_FEATURES
    )

    previous_gray = None

    previous_position = None
    smoothed_position = None

    trajectory = []
    debug_trajectory = []

    total_distance = 0.0
    max_speed = 0.0
    moving_time = 0.0

    # ============================================================
    # STOP TRACKING
    # ============================================================

    stop_count = 0
    longest_stop = 0.0

    # Number of consecutive stationary frames
    stationary_frames = 0

    # True only after the stationary period has been
    # confirmed as an actual stop
    is_stopped = False

    # Frame where the current stop started
    stop_start_frame = None

    # Require the ant to be stationary for this long
    # before calling it a stop.
    STOP_DURATION = 0.5

    STOP_CONFIRM_FRAMES = max(
        1,
        int(fps * STOP_DURATION)
    )

    lost_frames = 0

    # Candidate confirmation
    candidate_position = None
    candidate_frames = 0

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        debug_frame = frame.copy()

        # ========================================================
        # PREPROCESS
        # ========================================================

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        if previous_gray is None:

            previous_gray = gray

            cv2.imshow(
                "Ant Tracking Debug",
                debug_frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            continue

        # ========================================================
        # CAMERA MOTION
        # ========================================================

        camera_dx, camera_dy = (
            estimate_camera_motion(
                previous_gray,
                gray,
                orb
            )
        )

        # Stabilize current frame
        stabilized_gray = stabilize_frame(
            gray,
            camera_dx,
            camera_dy
        )

        # ========================================================
        # MOTION DETECTION
        # ========================================================

        difference = cv2.absdiff(
            previous_gray,
            stabilized_gray
        )

        _, motion_mask = cv2.threshold(
            difference,
            18,
            255,
            cv2.THRESH_BINARY
        )

        # Remove tiny noise
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3)
        )

        motion_mask = cv2.morphologyEx(
            motion_mask,
            cv2.MORPH_OPEN,
            kernel
        )

        # Connect nearby pixels
        motion_mask = cv2.morphologyEx(
            motion_mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        motion_mask = cv2.dilate(
            motion_mask,
            kernel,
            iterations=1
        )

        # ========================================================
        # FIND CONTOURS
        # ========================================================

        contours, _ = cv2.findContours(
            motion_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []

        frame_height, frame_width = gray.shape

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < MIN_AREA:
                continue

            if area > MAX_AREA:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            if w <= 0 or h <= 0:
                continue

            aspect_ratio = (
                max(w, h) /
                min(w, h)
            )

            # Reject extremely thin objects
            if aspect_ratio > 6:
                continue

            center_x = x + w / 2
            center_y = y + h / 2

            candidate = {
                "x": center_x,
                "y": center_y,
                "area": area,
                "w": w,
                "h": h,
                "aspect_ratio": aspect_ratio
            }

            candidate["score"] = (
                score_candidate(
                    candidate,
                    previous_position,
                    frame_width,
                    frame_height
                )
            )

            candidates.append(candidate)

        selected = None
        tracking = False

        # ========================================================
        # TRACKING MODE
        # ========================================================

        if previous_position is not None:

            nearby = []

            for candidate in candidates:

                distance = math.dist(
                    previous_position,
                    (
                        candidate["x"],
                        candidate["y"]
                    )
                )

                if distance <= MAX_JUMP:

                    nearby.append(
                        (
                            candidate["score"],
                            distance,
                            candidate
                        )
                    )

            # ----------------------------------------------------
            # Select highest scoring nearby candidate
            # ----------------------------------------------------

            if nearby:

                nearby.sort(
                    key=lambda item: item[0],
                    reverse=True
                )

                selected = nearby[0][2]

                current_position = (
                    selected["x"],
                    selected["y"]
                )

                lost_frames = 0
                tracking = True

            else:

                # ------------------------------------------------
                # No moving contour near ant.
                #
                # The ant may simply be standing still.
                # Keep the last position for a while.
                # ------------------------------------------------

                lost_frames += 1

                if lost_frames <= MAX_LOST_FRAMES:

                    current_position = (
                        previous_position[0],
                        previous_position[1]
                    )

                    tracking = True

                else:

                    current_position = None
                    tracking = False

        # ========================================================
        # SEARCH MODE
        # ========================================================

        else:

            current_position = None

            if candidates:

                candidates.sort(
                    key=lambda c: c["score"],
                    reverse=True
                )

                best = candidates[0]

                position = (
                    best["x"],
                    best["y"]
                )

                # ------------------------------------------------
                # Candidate confirmation
                # ------------------------------------------------

                if candidate_position is None:

                    candidate_position = position
                    candidate_frames = 1

                else:

                    distance = math.dist(
                        candidate_position,
                        position
                    )

                    if distance <= MAX_CANDIDATE_JUMP:

                        candidate_frames += 1

                        candidate_position = (
                            (
                                candidate_position[0]
                                + position[0]
                            ) / 2,

                            (
                                candidate_position[1]
                                + position[1]
                            ) / 2
                        )

                    else:

                        candidate_position = position
                        candidate_frames = 1

                # ------------------------------------------------
                # Draw candidate
                # ------------------------------------------------

                cv2.circle(
                    debug_frame,
                    (
                        int(candidate_position[0]),
                        int(candidate_position[1])
                    ),
                    8,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    debug_frame,
                    (
                        f"Candidate "
                        f"{candidate_frames}/"
                        f"{MIN_CONFIRM_FRAMES}"
                    ),
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

                # ------------------------------------------------
                # Confirm ant
                # ------------------------------------------------

                if candidate_frames >= MIN_CONFIRM_FRAMES:

                    current_position = (
                        candidate_position[0],
                        candidate_position[1]
                    )

                    previous_position = current_position

                    smoothed_position = current_position

                    candidate_position = None
                    candidate_frames = 0

                    lost_frames = 0
                    tracking = True

            else:

                candidate_position = None
                candidate_frames = 0

        # ========================================================
        # PROCESS TRACKED POSITION
        # ========================================================

        if (
            tracking
            and
            current_position is not None
        ):

            # ----------------------------------------------------
            # Smooth position
            # ----------------------------------------------------

            if smoothed_position is None:

                smoothed_position = current_position

            else:

                smoothed_position = (

                    SMOOTHING_ALPHA *
                    current_position[0]
                    +
                    (1 - SMOOTHING_ALPHA) *
                    smoothed_position[0],

                    SMOOTHING_ALPHA *
                    current_position[1]
                    +
                    (1 - SMOOTHING_ALPHA) *
                    smoothed_position[1]
                )

            position = smoothed_position

            # ----------------------------------------------------
            # Movement
            # ----------------------------------------------------

            movement = math.dist(
                previous_position,
                position
            )

            if movement >= MIN_MOVEMENT:

                # =================================================
                # ANT IS MOVING
                # =================================================

                total_distance += movement

                speed_pixels = (
                    movement * fps
                )

                max_speed = max(
                    max_speed,
                    speed_pixels
                )

                moving_time += 1 / fps

                # ------------------------------------------------
                # If we were stopped, the stop has ended.
                # Finalize its duration.
                # ------------------------------------------------

                if is_stopped:

                    if stop_start_frame is not None:

                        stop_duration = (
                            frame_number
                            - stop_start_frame
                        ) / fps

                        longest_stop = max(
                            longest_stop,
                            stop_duration
                        )

                    is_stopped = False
                    stop_start_frame = None

                # Reset stationary counter
                stationary_frames = 0

            else:

                # =================================================
                # ANT IS NOT MOVING
                # =================================================

                stationary_frames += 1

                # ------------------------------------------------
                # Only declare a stop after enough consecutive
                # stationary frames.
                # ------------------------------------------------

                if (
                    not is_stopped
                    and
                    stationary_frames >= STOP_CONFIRM_FRAMES
                ):

                    is_stopped = True

                    stop_count += 1

                    # Work backwards so the stop duration includes
                    # the frames during which we were confirming it.
                    stop_start_frame = (
                        frame_number
                        - stationary_frames
                        + 1
                    )

                # ------------------------------------------------
                # Update longest stop while still stopped
                # ------------------------------------------------

                if is_stopped and stop_start_frame is not None:

                    current_stop_duration = (
                        frame_number
                        - stop_start_frame
                        + 1
                    ) / fps

                    longest_stop = max(
                        longest_stop,
                        current_stop_duration
                    )

            # ----------------------------------------------------
            # Save trajectory
            # ----------------------------------------------------

            trajectory.append({
                "frame": frame_number,

                "time": round(
                    frame_number / fps,
                    3
                ),

                "x": round(
                    position[0],
                    2
                ),

                "y": round(
                    position[1],
                    2
                )
            })

            debug_trajectory.append(
                (
                    int(position[0]),
                    int(position[1])
                )
            )

            previous_position = position

        # ========================================================
        # DRAW TRAJECTORY
        # ========================================================

        for i in range(
            1,
            len(debug_trajectory)
        ):

            cv2.line(
                debug_frame,
                debug_trajectory[i - 1],
                debug_trajectory[i],
                (0, 255, 0),
                2
            )

        # ========================================================
        # DRAW ANT
        # ========================================================

        if (
            tracking
            and
            previous_position is not None
        ):

            ant_x = int(
                previous_position[0]
            )

            ant_y = int(
                previous_position[1]
            )

            cv2.circle(
                debug_frame,
                (
                    ant_x,
                    ant_y
                ),
                12,
                (0, 0, 255),
                2
            )

            cv2.circle(
                debug_frame,
                (
                    ant_x,
                    ant_y
                ),
                3,
                (0, 0, 255),
                -1
            )

        # ========================================================
        # STATUS
        # ========================================================

        if tracking:

            if is_stopped:

                status = "ANT ON BREAK"

                status_color = (
                    0,
                    255,
                    255
                )

            else:

                status = "TRACKING"

                status_color = (
                    0,
                    255,
                    0
                )

        elif lost_frames > 0:

            status = f"LOST {lost_frames}"

            status_color = (
                0,
                0,
                255
            )

        else:

            status = "SEARCHING"

            status_color = (
                0,
                255,
                255
            )

        # ========================================================
        # CURRENT SPEED
        # ========================================================

        current_speed = 0.0

        if len(debug_trajectory) >= 2:

            p1 = debug_trajectory[-2]
            p2 = debug_trajectory[-1]

            movement = math.dist(
                p1,
                p2
            )

            current_speed = (
                movement *
                fps /
                PIXELS_PER_ANT_METER
            )

        distance_meters = (
            total_distance /
            PIXELS_PER_ANT_METER
        )

        # ========================================================
        # DEBUG TEXT
        # ========================================================

        cv2.putText(
            debug_frame,
            status,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2
        )

        cv2.putText(
            debug_frame,
            (
                f"Distance: "
                f"{distance_meters:.2f} ant m"
            ),
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            debug_frame,
            (
                f"Speed: "
                f"{current_speed:.2f} ant m/s"
            ),
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            debug_frame,
            (
                f"Camera: "
                f"{camera_dx:.1f}, "
                f"{camera_dy:.1f}"
            ),
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            debug_frame,
            (
                f"Frame: "
                f"{frame_number}/"
                f"{total_frames}"
            ),
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            debug_frame,
            (
                f"Stops: "
                f"{stop_count}"
            ),
            (20, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # Show current stop duration
        if is_stopped and stop_start_frame is not None:

            current_stop_duration = (
                frame_number
                - stop_start_frame
                + 1
            ) / fps

            cv2.putText(
                debug_frame,
                (
                    f"Break: "
                    f"{current_stop_duration:.1f}s"
                ),
                (20, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

        else:

            cv2.putText(
                debug_frame,
                "Break: 0.0s",
                (20, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                2
            )

        cv2.putText(
            debug_frame,
            "Q = quit",
            (20, 270),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )

        # ========================================================
        # SHOW DEBUG WINDOWS
        # ========================================================

        cv2.imshow(
            "Ant Tracking Debug",
            debug_frame
        )

        cv2.imshow(
            "Motion Mask",
            motion_mask
        )

        # ========================================================
        # QUIT
        # ========================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        previous_gray = stabilized_gray

    # ============================================================
    # FINISH ANY STOP THAT CONTINUES UNTIL VIDEO END
    # ============================================================

    if (
        is_stopped
        and
        stop_start_frame is not None
    ):

        stop_duration = (
            frame_number
            - stop_start_frame
            + 1
        ) / fps

        longest_stop = max(
            longest_stop,
            stop_duration
        )

    # ============================================================
    # CLEANUP
    # ============================================================

    cap.release()

    cv2.destroyAllWindows()

    # ============================================================
    # STATISTICS
    # ============================================================

    distance = (
        total_distance /
        PIXELS_PER_ANT_METER
    )

    average_speed_pixels = (
        total_distance /
        moving_time
        if moving_time > 0
        else 0
    )

    average_speed = (
        average_speed_pixels /
        PIXELS_PER_ANT_METER
    )

    ant_max_speed = (
        max_speed /
        PIXELS_PER_ANT_METER
    )

    elapsed_time = (
        total_frames / fps
        if fps > 0
        else 0
    )

    estimated_steps = int(
        distance * 20
    )

    calories = round(
        distance * 0.8,
        1
    )

    if ant_max_speed > 0:

        terrain_score = int(
            (
                1 -
                average_speed /
                ant_max_speed
            ) * 100
        )

        terrain_score = max(
            0,
            min(
                100,
                terrain_score
            )
        )

    else:

        terrain_score = 0

    fitness_score = int(
        min(
            100,
            distance * 5 +
            average_speed * 10
        )
    )

    # ============================================================
    # ACHIEVEMENTS
    # ============================================================

    achievements = []

    if trajectory:

        achievements.append(
            "First Steps"
        )

    if distance >= 10:

        achievements.append(
            "10 AM Club"
        )

    if distance >= 25:

        achievements.append(
            "Marathon Ant"
        )

    if ant_max_speed >= 1:

        achievements.append(
            "Speed Demon"
        )

    if stop_count >= 10:

        achievements.append(
            "Professional Rest"
        )

    if moving_time >= 20:

        achievements.append(
            "Endurance Ant"
        )

    if not achievements:

        achievements.append(
            "Ant in Training"
        )

    # ============================================================
    # RESULT
    # ============================================================

    return {

        "title":
            generate_activity_title(
                distance,
                ant_max_speed,
                stop_count
            ),

        "description":
            generate_description(
                distance,
                moving_time
            ),

        "distance":
            round(distance, 2),

        "elapsed_time":
            round(elapsed_time, 2),

        "moving_time":
            round(moving_time, 2),

        "average_speed":
            round(average_speed, 2),

        "max_speed":
            round(ant_max_speed, 2),

        "stops":
            stop_count,

        "longest_stop":
            round(longest_stop, 2),

        "estimated_steps":
            estimated_steps,

        "calories":
            calories,

        "terrain_score":
            terrain_score,

        "fitness_score":
            fitness_score,

        "achievements":
            achievements,

        "trajectory":
            trajectory
    }

def generate_activity_title(
    distance,
    max_speed,
    stops
):

    if max_speed >= 2:
        return "THE GREAT ANT GRAND PRIX"

    if stops >= 10:
        return "A Very Productive Day"

    if distance >= 25:
        return "The Extremely Unnecessary Marathon"

    if distance >= 10:
        return "The Big Expedition"

    return "Suspicious Amount of Walking"


def generate_description(
    distance,
    moving_time
):

    if distance >= 25:

        return (
            "An absolutely unnecessary amount "
            "of walking. Scientists are concerned."
        )

    if moving_time < 5:

        return (
            "Short expedition. "
            "The ant probably forgot something."
        )

    return (
        "A completely legitimate athletic "
        "achievement that definitely deserved "
        "to be tracked."
    )