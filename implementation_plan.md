# Kế hoạch Thực thi KLTN (Updated Implementation Plan)

## 📌 Chủ đề Đề tài
**"Cải thiện độ chính xác đứng của mô hình địa hình số (DEM) bằng ICESat-2 và đánh giá tác động đến mô phỏng ngập lụt bãi sông Hồng, Hà Nội"**

---

## 🎯 1. Định hướng Cốt lõi & Sự Thay đổi Trọng tâm (Key Shift)

So với kế hoạch cũ tập trung chủ yếu vào mạng sinh đối kháng nhúng vật lý (PI-GAN) gây quá tải và phân tán, **kế hoạch mới tái định hình trọng tâm khoa học vào hai trụ cột chính**:

1. **Trụ cột 1 (Lõi Kỹ thuật - 60% khối lượng):** Xây dựng quy trình GeoAI nâng cấp, hiệu chỉnh sai số chiều đứng và downscale địa hình bãi sông Hồng từ FabDEM 30m thành **ANN-DTM 10m mặt đất thực (bare-earth)** bằng dữ liệu LiDAR vệ tinh ICESat-2 (ATL03/ATL08) và ảnh đa phổ Sentinel-2.
2. **Trụ cột 2 (Đánh giá Tác động & Ứng dụng - 40% khối lượng):** Đánh giá tác động lan truyền (Cascading Impact) của DTM 10m đã cải thiện đối với mô hình thủy lực 2D (HEC-RAS 2D / HAND) thông qua so sánh độ chênh lệch vết ngập, mực nước ngập, và ứng dụng đánh giá rủi ro cho **Trục đại lộ cảnh quan sông Hồng (11.418 ha, 16 xã/phường, 12 khu đất đối ứng BT và tuyến monorail 84 km)**.

---

## 🏗️ 2. Khung Phương pháp luận 6 Giai đoạn (Detailed Workflow)

```
[FabDEM 30m] + [Sentinel-2 10m] + [ICESat-2 LiDAR]
                   │
                   ▼
     [Feature Engineering (10 Features)]
                   │
                   ▼
   [GeoAI Residual Learning (ΔH = FabDEM - ICESat2)]
     ├── Baseline MLP
     ├── Residual MLP (Huber Loss)
     └── XGBoost / LightGBM Ensemble
                   │
                   ▼
     [ANN-DTM 10m Bare-Earth Product]
                   │
       ┌───────────┴───────────┐
       ▼                       ▼
 [FabDEM 30m Thô]       [ANN-DTM 10m Sạch]
       │                       │
       └───────────┬───────────┘
                   ▼
   [HEC-RAS 2D / HAND Flood Simulations]
                   │
                   ▼
 [Cascading Impact Evaluation (CSI, F1, FAR, Depth MAE)]
                   │
                   ▼
 [Assessment on Red River Axis (12 BT Plots, Monorail)]
```

---

### **Giai đoạn 1: Thu thập & Tiền xử lý Dữ liệu Viễn thám**
- **ICESat-2 (ATL03/ATL08):** Thu thập dải photon laser đo cao qua dải bãi sông Hồng bằng thư viện `icepyx` ([icepyx GitHub](https://github.com/icesat2py/icepyx)). Lọc photon mặt đất chất lượng cao (`signal_conf_ph >= 3`, loại bỏ nhiễu bằng cờ chất lượng).
- **FabDEM (30m):** Trích xuất DEM nền toàn cầu (đã lọc cây cối/nhà cửa cơ bản), resample về lưới 10m bằng phương pháp nội suy song tuyến tính (bilinear).
- **Sentinel-2 (10m):** Trích xuất ảnh đa phổ sạch mây (B2-Blue, B3-Green, B4-Red, B8-NIR) cùng thời điểm.

### **Giai đoạn 2: Trích xuất Tập Đặc trưng Không gian (Spatial Feature Engineering)**
Nâng cấp từ 5 features đơn sơ lên **10 đặc trưng đa chiều**:
1. **Phổ đa băng:** $B2, B3, B4, B8$ (dạng sóng phản xạ Sentinel-2).
2. **Chỉ số Thực vật:** $\text{NDVI} = \frac{B8 - B4}{B8 + B4}$ (lượng hóa tán thực vật bãi bồi gây dềnh cao độ).
3. **Chỉ số Nước:** $\text{MNDWI} = \frac{B3 - B11}{B3 + B11}$ (bóc tách ranh giới mép nước & bãi bồi ngập ẩm).
4. **Địa hình:** $\text{FabDEM}_{\text{elev}}$, $\text{Slope}$ (Độ dốc), $\text{Aspect}$ (Hướng sườn - nguồn gốc chính gây méo dạng vệt quét laser/radar).
5. **Thủy văn/Cấu trúc:** $\text{Dist2River}$ (Khoảng cách tới lòng dẫn sông Hồng - định vị vùng biến động sai số $< 1.2\text{ km}$).

### **Giai đoạn 3: Huấn luyện & Tối ưu hóa Mô hình DEM Corrector (`ann_dtm_corrector.py`)**
- **Bài toán:** Chuyển từ học cao độ tuyệt đối sang **Residual Learning (Học sai số chiều đứng)**:
  $$\Delta H = H_{\text{FabDEM}} - H_{\text{ICESat-2}}$$
  $$H_{\text{corrected}} = H_{\text{FabDEM}} - \Delta \hat{H}$$
- **Hàm Loss:** Thay MSE bằng **Huber Loss (Smooth L1 Loss)** nhằm hạn chế ảnh hưởng của nhiễu ngoại lệ (outliers) từ photon mặt trời.
- **Thử nghiệm Đa mô hình (Model Benchmarking):**
  - *Model A:* Baseline MLP (5 features, Direct Elevation - Code hiện tại).
  - *Model B:* Residual MLP (10 features, Huber Loss, Residual Learning).
  - *Model C:* XGBoost / LightGBM Regressor (10 features, Tree-based Ensemble).

### **Giai đoạn 4: Đánh giá & Kiểm định Học thuật DTM 10m**
- **Spatial K-Fold Cross-Validation:** Chia tập mẫu theo dải bay (Track-based Split) hoặc ô không gian (Spatial Clusters) để tránh rò rỉ dữ liệu không gian.
- **Phân tầng Đánh giá (Stratified Evaluation):** Lập bảng MAE, RMSE, $R^2$ phân chia theo:
  - Bãi bồi ven sông ($< 1.2\text{ km}$) vs. Vùng nông nghiệp vs. Khu dân cư đô thị.
  - Phân cấp độ dốc ($\text{Slope} < 2^\circ$ vs. $\text{Slope} \ge 2^\circ$).

### **Giai đoạn 5: Mô phỏng Thủy lực (HEC-RAS 2D / HAND) & Đánh giá Tác động Lan truyền**
- Thiết lập kịch bản trận lũ lịch sử trên sông Hồng (ví dụ: đỉnh lũ Sơn Tây / Long Biên).
- Chạy mô phỏng song song trên **02 mặt nền địa hình**:
  1. *Kịch bản Baseline:* Sử dụng **FabDEM 30m thô**.
  2. *Kịch bản Cải tiến:* Sử dụng **ANN-DTM 10m hiệu chỉnh**.
- **Chỉ số Đánh giá Tác động Lan truyền (Cascading Impact Metrics):**
  - **Độ trùng khớp vết ngập:** Critical Success Index (CSI), F1-Score, Probability of Detection (POD), False Alarm Ratio (FAR).
  - **Sai số độ sâu ngập:** MAE, RMSE của độ sâu mực nước ngập ($h$).
  - **So sánh vết ngập thực tế:** Kiểm chứng với ranh giới nước ngập bóc tách từ ảnh vệ tinh Ra-đa Sentinel-1 SAR.

### **Giai đoạn 6: Đánh giá Rủi ro Ngập lụt cho Trục Cảnh quan Sông Hồng**
- Chồng xếp bản đồ ngập siêu phân giải (10m) lên ranh giới:
  - **12 khu đất đối ứng hợp đồng BT** (5.073 ha).
  - **Tuyến đại lộ ven sông & tuyến Monorail 84 km**.
  - **16 xã, phường nghiên cứu**.
- Tính toán diện tích ngập, độ sâu ngập trung bình cho từng dự án thành phần và đề xuất khuyến nghị quy hoạch đô thị bọt biển (Sponge City) / thiết kế thích ứng.

---

## 📅 3. Lộ trình Thực hiện 10 Tuần (Updated Timeline)

| Tuần | Giai đoạn | Công việc Cụ thể | Sản phẩm Đầu ra |
| :--- | :--- | :--- | :--- |
| **1 - 2** | **1 & 2** | Tải ICESat-2, FabDEM, Sentinel-2. Tính 10 chỉ số đặc trưng (NDVI, MNDWI, Slope, Dist2River). | Dataset hoàn chỉnh (`data/processed/`) |
| **3 - 4** | **3** | Nâng cấp code `ann_dtm_corrector.py`: Residual learning, Huber Loss, huấn luyện MLP & XGBoost. | Mô hình DEM Corrector hoàn thiện |
| **5 - 6** | **4** | Xuất GeoTIFF **ANN-DTM 10m**. Kiểm định MAE/RMSE theo dải bay ICESat-2 và điểm GPS thực địa. | Bản đồ DTM 10m + Bảng kiểm định địa hình |
| **7 - 8** | **5** | Chạy mô phỏng ngập lụt HEC-RAS 2D / HAND trên FabDEM 30m vs ANN-DTM 10m. So sánh vết ngập SAR. | Bảng chỉ số CSI, F1, FAR & Bản đồ ngập so sánh |
| **9** | **6** | Phân tích rủi ro ngập cho 12 khu đất BT, tuyến Monorail và 16 xã/phường dọc sông Hồng. | Bản đồ & Bảng đánh giá rủi ro quy hoạch |
| **10 - 11**| **Báo cáo** | Hoàn thiện Thuyết minh Khóa luận, biểu đồ, hình vẽ và chuẩn bị slide bảo vệ. | Toàn văn Khóa luận Tốt nghiệp |

---

## 🧪 4. Khung Kiểm định Chất lượng (Validation Standards)

1. **Địa hình DTM 10m:**
   - MAE $< 0.30\text{ m}$, RMSE $< 0.45\text{ m}$ trên tập kiểm thử ICESat-2 độc lập.
   - Giảm sai số đứng $\ge 30\text{ cm}$ tại dải bãi bồi ven sông ($< 1.2\text{ km}$).
2. **Mô hình Ngập lụt (HEC-RAS 2D / HAND):**
   - Chỉ số trùng khớp vết ngập $\text{CSI} \ge 0.75$, $\text{F1-Score} \ge 0.80$.
   - Giảm tỷ lệ báo động giả ($\text{FAR}$) $\ge 15\%$ so với DEM 30m thô.
   - Vết ngập phù hợp với ranh giới lũ Sentinel-1 SAR thực tế.

---

## 🔗 5. Mã nguồn & Tài liệu Tham khảo (Code & Data References)

### **A. Mã nguồn Mở trực tiếp từ các Bài báo Chủ chốt**

1. **Coppo Frías et al. (2025)** – *Improving 2D hydraulic modelling in floodplain areas with ICESat-2 data* (Remote Sensing of Environment):
   - **Zenodo Repository (2025):** [https://doi.org/10.5281/zenodo.17070190](https://doi.org/10.5281/zenodo.17070190) *(Code xử lý dữ liệu ICESat-2 & ANN hiệu chỉnh DEM bãi sông)*.
   - **Zenodo Repository (2023):** [https://doi.org/10.5281/zenodo.6570492](https://doi.org/10.5281/zenodo.6570492) *(Code tích hợp ICESat-2 vào mô hình thủy lực 2D)*.
   - **DTU Dataset (Musaeus et al., 2024):** [https://doi.org/10.11583/dtu.24118023.v1](https://doi.org/10.11583/dtu.24118023.v1) *(Script trích xuất mặt cắt sông từ photon ATL03)*.

2. **Sun et al. (2026)** – *Evaluating the Vertical Accuracy of Global DEMs Using ICESat-2 and Its Cascading Impact on HAND-Based Flood Modeling* (Remote Sensing, MDPI):
   - **GitHub Repository:** [https://github.com/manyoarcane/HAND](https://github.com/manyoarcane/HAND) *(Python script lọc nhiễu photon ICESat-2 ATL08, chuyển đổi Geoid EGM96/EGM2008, tính chỉ số RMSE/MAE và mô phỏng ngập lụt HAND)*.

### **B. Hệ sinh thái Thư viện Mở dành cho ICESat-2 & DEM**

1. **`icepyx` (NASA ICESat-2 Community Library):**
   - **GitHub Repository:** [https://github.com/icesat2py/icepyx](https://github.com/icesat2py/icepyx)
   - *Chức năng:* Truy vấn, tải dữ liệu tự động theo Bounding Box sông Hồng, trích xuất biến số `ATL03`/`ATL08` nạp vào `GeoPandas`/`xarray`.

2. **`captoolkit` (NASA JPL Cryosphere Altimetry Processing Toolkit):**
   - **GitHub Repository:** [https://github.com/fspaolo/captoolkit](https://github.com/fspaolo/captoolkit)
   - *Chức năng:* Công cụ xử lý dải đo cao laser, nội suy lưới địa hình (gridding), tính `Slope`/`Aspect` và hiệu chỉnh sai số địa hình.

3. **`SlideRule` (NASA / UW eScience Institute):**
   - **Website & GitHub:** [https://slideruleearth.io](https://slideruleearth.io) | [https://github.com/SlideRuleEarth/sliderule-python](https://github.com/SlideRuleEarth/sliderule-python)
   - *Chức năng:* Điện toán đám mây cho phép trích xuất và lọc photon ICESat-2 siêu nhanh theo thời gian thực mà không cần tải các tệp HDF5 nặng về máy local.
