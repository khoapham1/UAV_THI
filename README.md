# Vision-Based UAV DropBall System

<p align="center">
  <strong>Autonomous UAV payload release using Hailo AI detection, Kalman filtering, and target-specific PID control.</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.x-blue">
  <img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-Computer%20Vision-green">
  <img alt="Hailo AI" src="https://img.shields.io/badge/Hailo-AI%20Inference-purple">
  <img alt="PID Control" src="https://img.shields.io/badge/PID-Closed--Loop%20Control-orange">
  <img alt="Kalman Filter" src="https://img.shields.io/badge/Kalman-Target%20Stabilization-red">
  <img alt="Platform" src="https://img.shields.io/badge/Platform-Raspberry%20Pi%20%2B%20Flight%20Controller-lightgrey">
</p>

---

## Overview

This project presents a **vision-based UAV DropBall system** for autonomous payload release missions.  
A downward-facing camera captures live frames, a **Hailo AI model** detects mission targets, and the UAV continuously adjusts its position using **closed-loop visual feedback**.

The system is designed for **precision aerial dropping**, where the UAV must align a detected target with a predefined **aim point** rather than the raw image center. This improves release accuracy by compensating for real drop geometry, camera offset, and mission-specific calibration.

The overall framework combines:

- **Real-time visual target detection**
- **Kalman-based target center stabilization**
- **Low-pass filtered image error**
- **Target-specific PID control**
- **Hold-zone validation before release**
- **Automatic payload drop logic**

This repository reflects a practical embedded UAV control pipeline for **vision-guided payload delivery** and can be extended to **precision landing** and **autonomous delivery missions**.

---

## Vietnamese Summary

Đây là dự án **UAV DropBall sử dụng xử lý ảnh** để thực hiện nhiệm vụ **thả vật chính xác theo mục tiêu quan sát được từ camera**.

Hệ thống kết hợp:

- **Hailo AI** để phát hiện mục tiêu theo thời gian thực
- **Kalman Filter** để ổn định tâm mục tiêu
- **Low-pass Filter** để làm mượt sai số ảnh
- **PID riêng cho từng target**
- **Aim point compensation** để tăng độ chính xác khi thả
- **Hold-zone logic** để xác nhận điều kiện release

Dự án hướng tới một hệ thống UAV tự hành có khả năng **detect → align → stabilize → release** theo vòng kín, phù hợp cho nghiên cứu về **visual servoing**, **payload delivery**, và **drone autonomy**.

---

## Key Features

- Real-time target detection using a **Hailo HEF model**
- Multi-class target recognition:
  - **Red**
  - **Yellow**
  - **Blue**
  - **H-marker**
- **Kalman filter** for target-center stabilization
- **Low-pass filter** for image-space error smoothing
- **Target-specific PID controllers**
- **Aim-point compensation** for precise payload release
- **Hold-zone logic** for release validation
- **Automatic DropBall trigger**
- PID logging and plotting for controller analysis
- Expandable framework for:
  - **visual landing**
  - **precision delivery**
  - **multi-target missions**

---

## System Architecture

```text
Downward Camera
      |
      v
Hailo AI Detection
      |
      v
Bounding Box + Target Center
      |
      v
Center Tracking + Kalman Filter
      |
      v
Aim Point Selection
      |
      v
Error Computation (ex, ey)
      |
      v
Low-Pass Filter
      |
      v
PID Controller
      |
      v
Velocity Command (vx, vy)
      |
      v
Flight Controller / UAV Motion
      |
      v
Camera Feedback
      ↺ Closed-Loop Visual Servoing
```

##Control Pipeline
- **1. Image Acquision**: The onboard camera continuously captures live frames from the downward-facing view.
- **2. AI-Based detection**:
    A Hailo neural inference model detects mission targets and returns:
    - **bounding box**
    - **confidence score**
    - **class label**
    - **target center**
- **3. Target Stabilization**
  To reduce jitter and temporary detection loss, the system applies:

    - **center memory / target tracking**
    - **Kalman filtering**
    - **low-pass filtering**

  This generates a more stable target estimate before control action.
- **4. Aim-Point Compensation**
  Each mission target is assigned its own aim point instead of using the raw image center.

This is important because the optimal release point is not always the exact center of the frame.
The calibrated aim point improves real-world drop accuracy.
<img width="1062" height="660" alt="image" src="https://github.com/user-attachments/assets/8e44e682-6338-4d7d-bff7-5b3f4602d348" />

- **5. Error Computation**
  For each detected target center (tx, ty) and calibrated aim point (aim_cx, aim_cy):
```text
ex = tx - aim_cx
ey = ty - aim_cy
```

These image-space errors represent how far the target is from the desired payload-release position.
  <img width="680" height="624" alt="image" src="https://github.com/user-attachments/assets/81d6a634-217c-4a1d-b481-7877cc157005" />
 
- **6. PID-Based Motion Control**
- The filtered errors are converted into UAV body-frame velocity commands:

  - **vx is derived from vertical image error**
  - **vy is derived from horizontal image error**


  <img width="679" height="199" alt="image" src="https://github.com/user-attachments/assets/c1cff190-9588-4c5f-9507-0197a4cd903d" />

- **7. Ball Release Logic**
- **8. Closed-Loop Feadback**
![download (2)](https://github.com/user-attachments/assets/d0143ca6-3646-4e57-8134-0c17c7f05a19)

## Mission Flow
The mission is executed as a sequence of autonomous targeting steps:

- **Arm and take off**
- **Search for the designated target**
- **Detect and confirm the correct object**
- **Align the UAV using PID control**
- **Enter the hold zone**
- **Trigger ball release**
- **Continue to the next target**
- **Perform final landing or visual landing stage**
## Author
- This repository reflects a practical engineering approach to vision-based UAV precision drop missions, combining real-time perception, estimation, and closed-loop control into a unified embedded system.

- If you are working on:

- **UAV control**
- **visual servoing**
- **payload delivery**
- **AI-assisted drone autonomy**

- this project can serve as a solid reference for further development.
