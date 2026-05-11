# OSC 数据协议

除非特别说明，本程序发送的所有空间坐标（X, Y, Z）均为归一化后的浮点数（范围通常为 `0.0` 到 `1.0`）。如果你需要获取屏幕上的实际像素坐标，只需将这些坐标值分别乘以接收端画面的真实宽度和高度即可。

## 帧信息 (Frame)

在每一帧画面处理完成并开始发送数据时，会首先发送该帧的基础信息。

```text
/mp/frame [timestamp_ms: int] [width: int] [height: int] [source_name: str]
```
如果引擎在处理该帧时抛出异常错误，则会发送错误信息：
```text
/mp/error [message: str]
```

## 身体骨骼 (Pose)

```text
/mp/pose/count [count: int]
/mp/pose/landmark [person_index: int] [landmark_index: int] [x: float] [y: float] [z: float] [visibility: float] [presence: float]
/mp/pose/world_landmark [person_index: int] [landmark_index: int] [x: float] [y: float] [z: float] [visibility: float] [presence: float]
```
*注：当前版本的 MediaPipe 骨骼追踪模块仅支持输出 1 个人（即 count 永远为 1，person_index 永远为 0）。其中 `world_landmark` 表示真实世界的三维坐标系，其单位为米（m）。*

## 手部追踪 (Hands)

```text
/mp/hands/count [count: int]
/mp/hands/handedness [hand_index: int] [label: str] [score: float]
/mp/hands/landmark [hand_index: int] [landmark_index: int] [x: float] [y: float] [z: float] [visibility: float] [presence: float]
```
*注：`label` 通常返回字符串 `Left`（左手）或 `Right`（右手）。程序最多支持同时追踪画面内的两只手。*

## 面部网格 (Face Mesh)

```text
/mp/face_mesh/count [count: int]
/mp/face_mesh/landmark [face_index: int] [landmark_index: int] [x: float] [y: float] [z: float] [visibility: float] [presence: float]
```
*注：该模块会输出一张脸上多达 468 个极其细腻的三维特征点坐标。*

## 人脸检测 (Face Detection)

```text
/mp/face_detection/count [count: int]
/mp/face_detection/box [face_index: int] [score: float] [xmin: float] [ymin: float] [width: float] [height: float]
```
*注：输出的检测框（box）所有四个参数均是相对于当前输入画面的归一化数值。*

## 自拍抠像 (Selfie Segmentation)

```text
/mp/selfie_segmentation/foreground_coverage [value: float]
```
*注：`value` 的范围是 `0.0` 到 `1.0`，它代表当前画面中被 AI 判定为“人（前景）”的像素，占整个画面总像素数量的百分比。你可以用这个数值在其他软件中制作“当人走出画面时触发事件”等交互逻辑。*
