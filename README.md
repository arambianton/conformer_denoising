# Conformer для задачи денойзинга речи

Задание по удалению шума из аудиосигналов с помощью архитектуры Conformer. Модель предсказывает маску для STFT-спектра зашумлённого сигнала и восстанавливает чистую речь через ISTFT.

**Данные:** LibriSpeech train-clean-100 (английские аудиокниги) + MUSAN (шумы) + русскоязычный корпус ruls_data для OOD-тестирования. Общий объём >15 ГБ.

**Архитектура:** STFT → Power Spectrogram → LayerNorm → Linear → 6× Conformer Block → Linear → Sigmoid → маскированный STFT → ISTFT. Итого **25 874 435** параметров.

Conformer реализован с нуля: Feed Forward, Depthwise Separable Convolution, Multi-Head Self-Attention с RPE.

**Обучение:** L1-loss на waveform, Adam + Noam Scheduler, gradient clipping, Mixed Precision.

**Результат:** SNR_model ≥ 11 dБ на всех 4 тестовых выборках.

**Стек:** `torch`, `torchaudio`, `omegaconf`, `tensorboard`
