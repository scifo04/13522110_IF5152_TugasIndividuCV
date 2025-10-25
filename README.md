# Tugas 1 Computer Vision IF5152

Deskripsi singkat konsep
1. Filtering: operasi pada citra untuk mereduksi noise atau menonjolkan detail. Contoh: mean (average), Gaussian (σ sebagai kontrol blur), median (bagus untuk salt-and-pepper), bilateral (preserve edges).
2. Edge detection: menemukan perubahan intensitas tajam — contoh Sobel, Prewitt, dan Canny. Canny umum karena pipeline smoothing → gradient → non-maximum suppression → hysteresis.
3. Feature point detection:
   a. Harris: deteksi sudut; sensitif pada perubahan intensitas dua arah.
   b. SIFT: scale- dan rotation-invariant; membangun scale-space (Gaussian dengan berbagai σ), mendeteksi extrema DoG, memberi descriptor (128-dim).
   c. FAST: sangat cepat; tes intensitas pada lingkaran pixel; biasanya dikombinasikan dengan deskriptor (ex: ORB).
4. Camera calibration: estimasi parameter kamera:
   a. Intrinsic: kamera matrix (fx, fy, cx, cy).
   b. Distortion: koefisien distorsi lensa (k1, k2, p1, p2, k3...).
   c. Extrinsic: rotasi (R) & translasi (t) yang menentukan pose kamera relatif dunia 3D. Biasanya pakai papan catur / ChArUco / circle grid, lalu cv2.calibrateCamera().

Tahap Menjalankan Program:
1. Git Clone repositori ini (git clone https://github.com/scifo05/13522110_IF5152_TugasIndividuCV.git)
2. Install Requirements (pip install -r requirements.txt)
3. Jalankan program (Pastikan dengan format ini: "python feature_folder/program_name.py". Contoh: "python 03_feature_points/code.py". Hal ini dilakukan supaya gambar bisa dibaca oleh program)
