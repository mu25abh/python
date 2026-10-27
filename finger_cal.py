# ============================================================
# Step 1 — IMPORTS
# ============================================================
import cv2
import mediapipe as mp
from collections import deque   # Step 15 (history)


# ============================================================
# Step 5 — Count non-thumb fingers
# ============================================================
def count_fingers_basic(landmarks_xy):
    """
    Count extended fingers among index, middle, ring, pinky.
    Rule: fingertip_y < pip_y → finger is open.
    """
    if landmarks_xy is None or len(landmarks_xy) != 21:
        return None

    count = 0
    tip_ids = [8, 12, 16, 20]
    pip_ids = [6, 10, 14, 18]

    for tip, pip in zip(tip_ids, pip_ids):
        tip_y = landmarks_xy[tip][1]
        pip_y = landmarks_xy[pip][1]
        if tip_y < pip_y:
            count += 1

    return count


# ============================================================
# Step 8 — Progress Bar
# ============================================================
def draw_progress_bar(img, progress, total, x=10, y=200, w=250, h=20,
                      fg=(0,200,0), bg=(50,50,50), border=(255,255,255)):

    cv2.rectangle(img, (x, y), (x+w, y+h), border, 2)
    fill_w = int(w * (progress / max(1, total)))
    cv2.rectangle(img, (x+2, y+2), (x+fill_w-2, y+h-2), fg, -1)


# ============================================================
# Step 1 — MAIN PROGRAM
# ============================================================
def main():

    # Step 1a — camera activation
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera Not Found.")
        return

    # Step 3a — MediaPipe Helpers
    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    # Step 7a — stability system
    STABLE_FRAMES_REQUIRED = 15
    stable_frames = 0
    last_count = None

    # Step 10a — operator
    operator = None

    # Step 11a — number and result
    first_num = None
    second_num = None
    result_value = None
    error_message = None

    # Step 15a — history store (5 calculations)
    history = deque(maxlen=5)

    # Step 3b — MediaPipe Hands Context
    with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        while True:

            ok, frame = cap.read()
            if not ok:
                break

            # Step 2 — Flip + Title
            frame = cv2.flip(frame, 1)

            cv2.putText(frame, "Press Q to Quit", (10,25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255,150,0), 2)

            # Step 3c — Hand Detection
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            # Read key
            key = cv2.waitKey(1) & 0xFF

            # Step 9b — ACCEPT KEY
            accept_pressed = (key == 13 or key == 32)  # Enter/Space

            # Step 10b — OPERATOR KEY
            if key in [ord('+'), ord('-'), ord('*'), ord('/')]:
                operator = chr(key)

            # Step 12 — RESET KEY
            if key == ord('r'):
                first_num = None
                second_num = None
                operator = None
                result_value = None
                error_message = None
                last_count = None
                stable_frames = 0

            # Step 15b — HISTORY KEY
            history_pressed = (key == ord('h'))

            # If hand exists:
            if result.multi_hand_landmarks:

                # 6a — Handedness
                hand_lm = result.multi_hand_landmarks[0]
                handed_label = None

                if result.multi_handedness:
                    handed_label = result.multi_handedness[0].classification[0].label

                # Step 4a — 21 landmarks
                pts_xy = [(lm.x, lm.y) for lm in hand_lm.landmark]

                # Step 4b — fingertip dot (yellow)
                h, w = frame.shape[:2]
                ix = int(pts_xy[8][0] * w)
                iy = int(pts_xy[8][1] * h)
                cv2.circle(frame, (ix,iy), 6, (0,255,255), -1)

                # Step 5b — non-thumb count
                current_count = count_fingers_basic(pts_xy)

                # Step 6b — thumb logic
                tip4_x = pts_xy[4][0]
                ip3_x  = pts_xy[3][0]

                if handed_label == "Right":
                    if tip4_x < ip3_x:
                        current_count += 1
                else:
                    if tip4_x > ip3_x:
                        current_count += 1

                # Step 7b — stability
                if current_count == last_count:
                    stable_frames = min(STABLE_FRAMES_REQUIRED, stable_frames + 1)
                else:
                    stable_frames = 1

                last_count = current_count

                # Step 11b — Accept FIRST or SECOND number
                if accept_pressed and stable_frames >= STABLE_FRAMES_REQUIRED:

                    if first_num is None:
                        first_num = current_count

                    elif operator is not None and second_num is None:
                        second_num = current_count

                # Step 11c — Compute Result
                result_value = None
                error_message = None

                if first_num is not None and operator is not None and second_num is not None:

                    try:
                        if operator == '+':
                            result_value = first_num + second_num
                        elif operator == '-':
                            result_value = first_num - second_num
                        elif operator == '*':
                            result_value = first_num * second_num
                        elif operator == '/':
                            if second_num == 0:
                                raise ZeroDivisionError
                            result_value = first_num / second_num

                    except ZeroDivisionError:
                        error_message = "Cannot divide by zero"

                    # Step 15c — Add to history
                    if result_value is not None:
                        history.appendleft(f"{first_num} {operator} {second_num} = {result_value}")

                # ------------------------
                # UI TEXT (Improved Colors)
                # ------------------------

                # Count (cyan)
                cv2.putText(frame, f"Count: {current_count}", (10,100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

                # Stability (green)
                cv2.putText(frame, f"Stable: {stable_frames}/{STABLE_FRAMES_REQUIRED}",
                            (10,140), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0,255,0), 2)

                # Progress Bar
                draw_progress_bar(frame, stable_frames, STABLE_FRAMES_REQUIRED)

                # Step-by-step guidance (IMPORTANT)
                if first_num is None:
                    cv2.putText(frame,
                        "Show a number & hold steady, then press ENTER/SPACE to accept",
                        (10,180), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255,255,0), 2)

                elif first_num is not None and operator is None:
                    cv2.putText(frame,
                        "Now press +  -  *  /  to select an operator",
                        (10,180), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0,255,255), 2)

                elif first_num is not None and operator is not None and second_num is None:
                    cv2.putText(frame,
                        "Now show second number & press ENTER/SPACE to accept",
                        (10,180), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0,200,255), 2)

                # Step 10c — Operator display
                cv2.putText(frame, f"Operator: {operator}", (10,260),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,200,0), 2)

                # Step 11d — First / Second
                cv2.putText(frame, f"First: {first_num}", (10,300),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,150), 2)

                cv2.putText(frame, f"Second: {second_num}", (10,340),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,150), 2)

                # Result
                if error_message:
                    cv2.putText(frame, error_message, (10,380),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

                elif result_value is not None:
                    cv2.putText(frame, f"Result: {result_value}", (10,380),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0,255,255), 2)

                # Reset instructions
                cv2.putText(frame, "Press R to Reset", (10,430),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,150,255), 2)

                # Step 15d — show history on 'H'
                if history_pressed:
                    cv2.putText(frame, "Last 5 Calculations:", (10,470),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,200,200), 2)
                    y = 510
                    for hline in history:
                        cv2.putText(frame, hline, (10, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (255,255,0), 2)
                        y += 40

                # Draw landmarks
                mp_draw.draw_landmarks(frame, hand_lm,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style())

            # Show final frame
            cv2.imshow("MyFingerCalc", frame)

            # exit
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# Step 1c — run main
# ============================================================
if __name__ == "__main__":
    main()