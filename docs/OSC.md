# OSC 数据协议 (ofxMediaPipe 兼容格式)

本项目发送的 OSC 数据遵循 [ofxMediaPipePython](https://github.com/design-io/ofxMediaPipePython) 标准格式，方便直接接入 OpenFrameworks, TouchDesigner, Unreal Engine 等创意编程环境。

## 核心设计
1. **归一化坐标**：除非特别说明，所有空间坐标（X, Y, Z）均为 `0.0` 到 `1.0` 的归一化浮点数。
2. **扁平化列表 (Flattened List)**：同一对象（如一只手）的所有关键点坐标会打包在一条消息中发送，而不是每个点发一条。
3. **帧同步 (Frame Sync)**：每条消息的第一个参数均为 **Frame Number**（帧编号），用于在接收端同步不同模型的数据。
4. **唯一标识 (ID)**：每条消息的第二个参数为对象 **ID**（如第几只手）。

---

## 基础元数据 (Metadata)

### 视频分辨率
每帧开始时发送，用于接收端进行坐标映射：
```text
/ofxmp/frame/video [x: 0] [y: 0] [width: int] [height: int]
```

### 追踪区域 (ROI)
```text
/ofxmp/frame/faces [x] [y] [w] [h]
/ofxmp/frame/hands [x] [y] [w] [h]
/ofxmp/frame/poses [x] [y] [w] [h]
```

---

## 身体骨骼 (Pose)

```text
/ofxmp/poses [FrameNum: int] [ID: int] [x0, y0, z0, x1, y1, z1, ...]
/ofxmp/posesW [FrameNum: int] [ID: int] [x0, y0, z0, x1, y1, z1, ...]
```
- **关键点数量**：33 个（总共 99 个浮点坐标）。
- **W 后缀**：表示 World Landmarks（真实世界三维坐标，单位为米）。
- **多人支持**：当前版本主要支持 1 人追踪，但已预留多人数据接口。

---

## 手部追踪 (Hands)

```text
/ofxmp/hands [FrameNum: int] [ID: int] [x0, y0, z0, x1, y1, z1, ...]
/ofxmp/hand/label [FrameNum: int] [ID: int] [label: str]
/ofxmp/hand/score [FrameNum: int] [ID: int] [score: float]
```
- **关键点数量**：21 个（总共 63 个浮点坐标）。
- **多人支持**：支持同时追踪画面中多达 **4** 只手。
- **label**：`Left` 或 `Right`。

---

## 面部网格 (Face Mesh)

```text
/ofxmp/faces [FrameNum: int] [ID: int] [x0, y0, z0, x1, y1, z1, ...]
```
- **关键点数量**：468 个（总共 1404 个浮点坐标）。
- **多人支持**：支持同时追踪多达 **4** 张脸。

---

## 人脸检测 (Face Detection)

```text
/ofxmp/face/box [FrameNum: int] [ID: int] [score: float] [xmin] [ymin] [w] [h]
```

---

## 自拍抠像 (Selfie Segmentation)

```text
/ofxmp/selfie_segmentation/foreground_coverage [value: float]
```
- `value`: 前景像素占比 (0.0 - 1.0)。
