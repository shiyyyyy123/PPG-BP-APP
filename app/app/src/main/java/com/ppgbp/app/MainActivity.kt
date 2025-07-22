package com.ppgbp.app

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.github.mikephil.charting.components.XAxis
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import com.ppgbp.app.databinding.ActivityMainBinding
import java.nio.ByteBuffer
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

// 新增特征提取器类
class FeatureExtractor {
    companion object {
        // 从PPG信号计算一阶导数(VPG)
        fun calculateVPG(ppg: List<Float>): List<Float> {
            val vpg = mutableListOf<Float>()
            for (i in 1 until ppg.size) {
                vpg.add(ppg[i] - ppg[i-1])
            }
            return vpg
        }

        // 从VPG信号计算二阶导数(APG)
        fun calculateAPG(vpg: List<Float>): List<Float> {
            val apg = mutableListOf<Float>()
            for (i in 1 until vpg.size) {
                apg.add(vpg[i] - vpg[i-1])
            }
            return apg
        }

        // 计算信号的直方图特征
        fun calculateHistogramFeatures(signal: List<Float>, numBins: Int = 10): Pair<List<Float>, List<Float>> {
            if (signal.isEmpty()) return Pair(listOf(), listOf())

            val min = signal.minOrNull() ?: 0f
            val max = signal.maxOrNull() ?: 0f
            val range = max - min
            val binSize = if (range > 0) range / numBins else 1f

            val upHistogram = MutableList(numBins) { 0f }
            val downHistogram = MutableList(numBins) { 0f }

            for (i in 1 until signal.size) {
                val value = signal[i]
                val prevValue = signal[i-1]
                val binIndex = min(numBins - 1, ((value - min) / binSize).toInt())

                if (value > prevValue) {
                    upHistogram[binIndex]++
                } else if (value < prevValue) {
                    downHistogram[binIndex]++
                }
            }

            // 归一化直方图
            val totalUp = upHistogram.sum()
            val totalDown = downHistogram.sum()

            val normalizedUp = if (totalUp > 0) upHistogram.map { it / totalUp } else upHistogram
            val normalizedDown = if (totalDown > 0) downHistogram.map { it / totalDown } else downHistogram

            return Pair(normalizedUp, normalizedDown)
        }

        // 计算邻域最大/最小值特征
        fun calculateNeighborExtremumFeatures(signal: List<Float>, windowSize: Int = 5): Pair<Float, Float> {
            if (signal.size <= windowSize) return Pair(0f, 0f)

            var maxNeighborMean = 0f
            var minNeighborMean = 0f
            var maxCount = 0
            var minCount = 0

            for (i in windowSize until signal.size - windowSize) {
                val window = signal.subList(i - windowSize, i + windowSize + 1)
                val max = window.maxOrNull() ?: 0f
                val min = window.minOrNull() ?: 0f

                if (signal[i] == max) {
                    maxNeighborMean += window.average().toFloat()
                    maxCount++
                }

                if (signal[i] == min) {
                    minNeighborMean += window.average().toFloat()
                    minCount++
                }
            }

            maxNeighborMean = if (maxCount > 0) maxNeighborMean / maxCount else 0f
            minNeighborMean = if (minCount > 0) minNeighborMean / minCount else 0f

            return Pair(maxNeighborMean, minNeighborMean)
        }

        // 检测PPG波形的特征点 (a, b, c, d, e 点)
        fun detectAPGPoints(ppg: List<Float>, vpg: List<Float>, apg: List<Float>): Map<String, Float> {
            val result = mutableMapOf<String, Float>()

            // 寻找收缩期开始点
            val peaks = mutableListOf<Int>()
            for (i in 2 until ppg.size - 2) {
                if (ppg[i] > ppg[i - 1] && ppg[i] > ppg[i + 1] &&
                    ppg[i] > ppg[i - 2] && ppg[i] > ppg[i + 2]) {
                    peaks.add(i)
                }
            }

            if (peaks.isEmpty()) return result

            // 对每个心动周期计算特征点
            val points = mutableMapOf<String, MutableList<Float>>()
            val timings = mutableMapOf<String, MutableList<Float>>()

            for (peakIndex in peaks) {
                if (peakIndex < 10 || peakIndex >= apg.size - 10) continue

                var aPoint: Float
                var bPoint = 0f
                var cPoint = 0f
                var dPoint = 0f
                var ePoint = 0f

                var aIndex = peakIndex
                var bIndex = -1
                var cIndex = -1
                var dIndex = -1
                var eIndex = -1

                // a点是VPG的正峰值
                aPoint = vpg[aIndex]

                // 在a之后查找b点(第一个负极值)
                for (i in aIndex + 1 until min(aIndex + 20, vpg.size)) {
                    if (i < vpg.size - 1 &&
                        vpg[i] < vpg[i - 1] && vpg[i] < vpg[i + 1]) {
                        bIndex = i
                        bPoint = vpg[i]
                        break
                    }
                }

                // 在b之后查找c点(第二个正极值)
                if (bIndex > 0) {
                    for (i in bIndex + 1 until min(bIndex + 20, vpg.size)) {
                        if (i < vpg.size - 1 &&
                            vpg[i] > vpg[i - 1] && vpg[i] > vpg[i + 1]) {
                            cIndex = i
                            cPoint = vpg[i]
                            break
                        }
                    }
                }

                // 在c之后查找d点(第二个负极值)
                if (cIndex > 0) {
                    for (i in cIndex + 1 until min(cIndex + 20, vpg.size)) {
                        if (i < vpg.size - 1 &&
                            vpg[i] < vpg[i - 1] && vpg[i] < vpg[i + 1]) {
                            dIndex = i
                            dPoint = vpg[i]
                            break
                        }
                    }
                }

                // 在d之后查找e点(第三个正极值)
                if (dIndex > 0) {
                    for (i in dIndex + 1 until min(dIndex + 20, vpg.size)) {
                        if (i < vpg.size - 1 &&
                            vpg[i] > vpg[i - 1] && vpg[i] > vpg[i + 1]) {
                            eIndex = i
                            ePoint = vpg[i]
                            break
                        }
                    }
                }

                // 保存点值
                points.getOrPut("a") { mutableListOf() }.add(aPoint)
                points.getOrPut("b") { mutableListOf() }.add(bPoint)
                points.getOrPut("c") { mutableListOf() }.add(cPoint)
                points.getOrPut("d") { mutableListOf() }.add(dPoint)
                points.getOrPut("e") { mutableListOf() }.add(ePoint)

                // 保存时间点 (相对于a点的距离)
                if (bIndex > 0) timings.getOrPut("b") { mutableListOf() }.add((bIndex - aIndex).toFloat())
                if (cIndex > 0) timings.getOrPut("c") { mutableListOf() }.add((cIndex - aIndex).toFloat())
                if (dIndex > 0) timings.getOrPut("d") { mutableListOf() }.add((dIndex - aIndex).toFloat())
                if (eIndex > 0) timings.getOrPut("e") { mutableListOf() }.add((eIndex - aIndex).toFloat())
            }

            // 计算平均值
            val apgA = points["a"]?.average()?.toFloat() ?: 0f
            val apgB = points["b"]?.average()?.toFloat() ?: 0f
            val apgC = points["c"]?.average()?.toFloat() ?: 0f
            val apgD = points["d"]?.average()?.toFloat() ?: 0f
            val apgE = points["e"]?.average()?.toFloat() ?: 0f

            result["apg_a"] = apgA
            result["apg_b"] = apgB
            result["apg_c"] = apgC
            result["apg_d"] = apgD
            result["apg_e"] = apgE

            // 计算比值特征
            if (apgA != 0f) {
                result["ratio_apg_b"] = apgB / apgA
                result["ratio_apg_c"] = apgC / apgA
                result["ratio_apg_d"] = apgD / apgA
                result["ratio_apg_e"] = apgE / apgA
            }

            // 时间特征
            val tB = timings["b"]?.average()?.toFloat() ?: 0f
            val tC = timings["c"]?.average()?.toFloat() ?: 0f
            val tD = timings["d"]?.average()?.toFloat() ?: 0f
            val tE = timings["e"]?.average()?.toFloat() ?: 0f

            result["T_b"] = tB
            result["T_c"] = tC
            result["T_d"] = tD
            result["T_e"] = tE

            // 心动周期平均长度
            val cycleDuration = if (peaks.size >= 2) {
                var sum = 0
                for (i in 1 until peaks.size) {
                    sum += peaks[i] - peaks[i-1]
                }
                sum.toFloat() / (peaks.size - 1)
            } else {
                0f
            }

            // 归一化时间特征
            if (cycleDuration > 0) {
                result["T_a_norm"] = 0f
                result["T_b_norm"] = tB / cycleDuration
                result["T_c_norm"] = tC / cycleDuration
                result["T_d_norm"] = tD / cycleDuration
                result["T_e_norm"] = tE / cycleDuration
            }

            // 计算其他特征
            val peakValues = peaks.map { ppg[it] }
            val avgPeakValue = peakValues.average().toFloat()

            // 波形峰值相对位置特征
            for (peakIndex in peaks) {
                if (peakIndex >= ppg.size - 1) continue

                // 找到峰值后的谷值点
                var valleyIndex = -1
                for (i in peakIndex + 1 until min(peakIndex + 50, ppg.size)) {
                    if (i < ppg.size - 1 &&
                        ppg[i] < ppg[i-1] && ppg[i] < ppg[i+1]) {
                        valleyIndex = i
                        break
                    }
                }

                if (valleyIndex > 0) {
                    val peakTime = peakIndex.toFloat()
                    val valleyTime = valleyIndex.toFloat()
                    val ts = valleyTime - peakTime
                    result["Ts"] = (result["Ts"] ?: 0f) + ts

                    if (cycleDuration > 0) {
                        result["Ts_norm"] = (result["Ts_norm"] ?: 0f) + (ts / cycleDuration)
                    }

                    // 计算收缩期和舒张期面积
                    val systolicArea = (0 until valleyIndex - peakIndex).sumOf {
                        (ppg[peakIndex + it] - ppg[valleyIndex]).toDouble()
                    }.toFloat()

                    result["AUCsys"] = (result["AUCsys"] ?: 0f) + systolicArea

                    if (avgPeakValue > 0) {
                        result["AUCsys_norm"] = (result["AUCsys_norm"] ?: 0f) + (systolicArea / avgPeakValue)
                    }
                }
            }

            // 平均值处理
            val peakCount = peaks.size.toFloat()
            if (peakCount > 0) {
                result["Ts"] = (result["Ts"] ?: 0f) / peakCount
                result["Ts_norm"] = (result["Ts_norm"] ?: 0f) / peakCount
                result["AUCsys"] = (result["AUCsys"] ?: 0f) / peakCount
                result["AUCsys_norm"] = (result["AUCsys_norm"] ?: 0f) / peakCount
            }

            // 计算血管弹性指数 Augmentation Index (AI)
            if (result["apg_d"] != null && result["apg_a"] != null && result["apg_a"] != 0f) {
                result["AI"] = result["apg_d"]!! / result["apg_a"]!!
            }

            return result
        }

        // 提取DSP特征
        fun extractTimeFeatures(ppg: List<Float>): Map<String, Float> {
            val result = mutableMapOf<String, Float>()

            // 检测峰值和谷值
            val peaks = mutableListOf<Int>()
            val valleys = mutableListOf<Int>()

            for (i in 2 until ppg.size - 2) {
                if (ppg[i] > ppg[i-1] && ppg[i] > ppg[i+1] &&
                    ppg[i] > ppg[i-2] && ppg[i] > ppg[i+2]) {
                    peaks.add(i)
                } else if (ppg[i] < ppg[i-1] && ppg[i] < ppg[i+1] &&
                    ppg[i] < ppg[i-2] && ppg[i] < ppg[i+2]) {
                    valleys.add(i)
                }
            }

            if (peaks.isEmpty() || valleys.isEmpty()) return result

            // 收缩时间、舒张时间和总时间
            var totalSysTime = 0f
            var totalDiaTime = 0f
            var totalSteepest = 0f
            var totalNegSteepest = 0f
            var count = 0

            for (peakIndex in peaks) {
                // 找到这个峰之前的谷
                var prevValley = -1
                for (v in valleys) {
                    if (v < peakIndex) prevValley = v
                    else break
                }

                // 找到这个峰之后的谷
                var nextValley = -1
                for (v in valleys) {
                    if (v > peakIndex) {
                        nextValley = v
                        break
                    }
                }

                if (prevValley >= 0 && nextValley >= 0) {
                    val sysTime = peakIndex - prevValley
                    val diaTime = nextValley - peakIndex

                    // 计算最陡上升和下降的斜率
                    var maxSlope = Float.MIN_VALUE
                    var minSlope = Float.MAX_VALUE

                    // 上升段斜率
                    for (i in prevValley until peakIndex) {
                        val slope = ppg[i+1] - ppg[i]
                        if (slope > maxSlope) maxSlope = slope
                    }

                    // 下降段斜率
                    for (i in peakIndex until nextValley) {
                        val slope = ppg[i+1] - ppg[i]
                        if (slope < minSlope) minSlope = slope
                    }

                    totalSysTime += sysTime
                    totalDiaTime += diaTime
                    totalSteepest += maxSlope
                    totalNegSteepest += abs(minSlope)
                    count++
                }
            }

            if (count > 0) {
                val avgCycleDuration = (totalSysTime + totalDiaTime) / count

                result["Td"] = totalSysTime / count
                result["Tsteepest"] = totalSteepest / count
                result["TNegSteepest"] = totalNegSteepest / count
                result["TdiaRise"] = totalDiaTime / count
                result["SteepDiaRise"] = totalSteepest / totalNegSteepest
                result["TSystoDiaRise"] = totalSysTime / totalDiaTime
                result["TdiaToEnd"] = totalDiaTime / count

                // 归一化特征
                result["Ts_norm"] = (totalSysTime / count) / avgCycleDuration
                result["TNegSteepest_norm"] = (totalNegSteepest / count) / totalSteepest
                result["TdiaRise_norm"] = (totalDiaTime / count) / avgCycleDuration
            }

            return result
        }
    }
}

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var isMonitoring = false

    // 使用固定大小的数组替代无限增长的列表，提高内存效率
    private val ppgSignalBuffer = FloatArray(1800) // 固定大小的数组
    private var ppgSignalBufferIndex = 0 // 当前写入位置
    private var ppgSignalBufferFull = false // 标记缓冲区是否已填满一轮

    private val displaySignalBuffer = mutableListOf<Entry>()
    private val ppgWindowSize = 1800
    private val displayWindowSize = 200
    private val requiredSampleLength = 1200

    private lateinit var bpPredictor: BloodPressurePredictor

    private var hasValidSignal = false
    private var collectionProgress = 0

    private var isResultLocked = false
    private var lockedSystolic = 0
    private var lockedDiastolic = 0

    private var signalStabilityCounter = 0
    private val requiredStableFrames = 30
    private val stabilityThreshold = 20f
    private var lastMean = 0f
    private var isSignalStable = false

    private val yAxisRange = 6f
    private var currentMeanValue = 75f
    private val smoothingFactor = 0.1f

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        if (allPermissionsGranted()) {
            setupCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }

        setupPpgChart()

        bpPredictor = BloodPressurePredictor(this)

        binding.controlButton.setOnClickListener {
            toggleMonitoring()
        }

        cameraExecutor = Executors.newSingleThreadExecutor()

        binding.systolicTextView.text = getString(R.string.systolic_pressure, 0)
        binding.diastolicTextView.text = getString(R.string.diastolic_pressure, 0)
        binding.progressTextView.text = getString(R.string.progress_text,0)

        // 添加特征维度信息日志
        Log.i(TAG, "Application initialized")
        Log.i(TAG, "Feature dimension: ${bpPredictor.getInputDimension()}")
        Log.i(TAG, "Model file path: ${this.filesDir}/student_mobile.onnx")
    }

    private fun setupPpgChart() {
        val chart = binding.ppgChart

        chart.description.isEnabled = false
        chart.legend.isEnabled = false
        chart.setTouchEnabled(false)
        chart.isDragEnabled = false
        chart.setScaleEnabled(false)
        chart.setPinchZoom(false)
        chart.setDrawGridBackground(false)

        val xAxis = chart.xAxis
        xAxis.position = XAxis.XAxisPosition.BOTTOM
        xAxis.setDrawGridLines(true)
        xAxis.textColor = Color.GRAY
        xAxis.setDrawAxisLine(true)
        xAxis.axisLineColor = Color.GRAY
        xAxis.setLabelCount(6, true)
        xAxis.valueFormatter = null

        xAxis.axisMinimum = 0f
        xAxis.axisMaximum = 200f

        val leftAxis = chart.axisLeft
        leftAxis.setDrawGridLines(true)
        leftAxis.textColor = Color.GRAY
        leftAxis.setDrawAxisLine(true)
        leftAxis.axisLineColor = Color.GRAY
        leftAxis.setLabelCount(6, true)

        updateYAxisRange()

        chart.axisRight.isEnabled = false

        val emptyDataSet = LineDataSet(ArrayList(), "PPG signal")
        emptyDataSet.color = Color.RED
        emptyDataSet.setDrawCircles(false)
        emptyDataSet.setDrawValues(false)
        emptyDataSet.mode = LineDataSet.Mode.CUBIC_BEZIER
        chart.data = LineData(emptyDataSet)

        chart.animateX(0)
    }

    private fun updateYAxisRange() {
        val leftAxis = binding.ppgChart.axisLeft
        leftAxis.axisMinimum = currentMeanValue - yAxisRange/2
        leftAxis.axisMaximum = currentMeanValue + yAxisRange/2
    }

    private fun setupCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(binding.cameraPreview.surfaceProvider)
                }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, PpgAnalyzer { redMean ->
                        processPpgSignal(redMean)
                    })
                }

            try {
                cameraProvider?.unbindAll()

                val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

                camera = cameraProvider?.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer
                )

                enableTorch(false)

            } catch (e: Exception) {
                Log.e(TAG, "Camera binding failed", e)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun toggleMonitoring() {
        if (isMonitoring) {
            stopMonitoring()
        } else {
            startMonitoring()
        }
    }

    private fun startMonitoring() {
        isMonitoring = true
        isResultLocked = false
        isSignalStable = false
        signalStabilityCounter = 0
        lastMean = 0f
        currentMeanValue = 75f

        // 重置缓冲区状态而不是创建新对象
        ppgSignalBufferIndex = 0
        ppgSignalBufferFull = false
        displaySignalBuffer.clear()

        updateYAxisRange()

        binding.controlButton.text = getString(R.string.stop_monitoring)
        binding.progressBar.progress = 0
        binding.progressTextView.text = getString(R.string.progress_text, 0)
        binding.systolicTextView.text = getString(R.string.systolic_pressure, 0)
        binding.diastolicTextView.text = getString(R.string.diastolic_pressure, 0)

        enableTorch(true)
    }

    private fun stopMonitoring() {
        isMonitoring = false

        binding.controlButton.text = getString(R.string.start_monitoring)

        enableTorch(false)
    }

    private fun enableTorch(enable: Boolean) {
        camera?.let {
            if (it.cameraInfo.hasFlashUnit()) {
                it.cameraControl.enableTorch(enable)
            }
        }
    }

    private fun processPpgSignal(redMean: Float) {
        if (!isMonitoring || isResultLocked) return

        val isFingerDetected = redMean > 50.0f

        if (isFingerDetected) {
            hasValidSignal = true

            if (!isSignalStable) {
                val signalDiff = abs(redMean - lastMean)
                if (signalDiff < stabilityThreshold) {
                    signalStabilityCounter++
                    if (signalStabilityCounter >= requiredStableFrames) {
                        isSignalStable = true

                        // 重置缓冲区状态而不是清空列表
                        ppgSignalBufferIndex = 0
                        ppgSignalBufferFull = false
                        displaySignalBuffer.clear()

                        currentMeanValue = redMean
                        updateYAxisRange()
                        
                        // 添加日志，表明信号已稳定
                        Log.d(TAG, "Signal is stable, starting data collection")
                    }
                } else {
                    signalStabilityCounter = 0
                }
                lastMean = redMean
                binding.cameraStatusTextView.text = "Waiting for signal stability..."
                return
            }

            binding.cameraStatusTextView.text = getString(R.string.measuring_text)

            // 使用循环缓冲区模式写入数据
            ppgSignalBuffer[ppgSignalBufferIndex] = redMean
            ppgSignalBufferIndex = (ppgSignalBufferIndex + 1) % ppgWindowSize
            if (ppgSignalBufferIndex == 0) {
                ppgSignalBufferFull = true
                // 添加日志，表明缓冲区已填满一轮
                Log.d(TAG, "Buffer is full")
            }

            currentMeanValue = currentMeanValue * (1 - smoothingFactor) + redMean * smoothingFactor
            updateYAxisRange()

            updateSignalChart()

            updateProgress()

            if (getEffectiveBufferSize() >= requiredSampleLength && collectionProgress >= 100) {
                Log.d(TAG, "Data collection complete, starting blood pressure prediction")
                predictBloodPressure()
            }
        } else {
            hasValidSignal = false
            isSignalStable = false
            signalStabilityCounter = 0
            binding.cameraStatusTextView.text = getString(R.string.no_finger_detected)
        }
    }

    private fun updateSignalChart() {
        // 清空显示缓冲区
        displaySignalBuffer.clear()

        // 计算要显示的数据起始位置
        val effectiveSize = getEffectiveBufferSize()
        val displayCount = min(effectiveSize, displayWindowSize)

        if (displayCount > 0) {
            if (ppgSignalBufferFull) {
                // 缓冲区已满，显示最近的displayCount个样本
                val startIdx = (ppgSignalBufferIndex + ppgWindowSize - displayCount) % ppgWindowSize
                for (i in 0 until displayCount) {
                    val idx = (startIdx + i) % ppgWindowSize
                    val x = i.toFloat()
                    displaySignalBuffer.add(Entry(x, ppgSignalBuffer[idx]))
                }
            } else {
                // 缓冲区未满
                val startIdx = max(0, ppgSignalBufferIndex - displayCount)
                for (i in 0 until displayCount) {
                    val idx = startIdx + i
                    val x = i.toFloat()
                    displaySignalBuffer.add(Entry(x, ppgSignalBuffer[idx]))
                }
            }
        }

        val dataSet = LineDataSet(displaySignalBuffer, "PPG signal")
        dataSet.color = Color.RED
        dataSet.setDrawCircles(false)
        dataSet.setDrawValues(false)
        dataSet.lineWidth = 2f
        dataSet.mode = LineDataSet.Mode.CUBIC_BEZIER

        val lineData = LineData(dataSet)
        binding.ppgChart.data = lineData

        binding.ppgChart.notifyDataSetChanged()
        binding.ppgChart.invalidate()
    }

    private fun updateProgress() {
        collectionProgress = min(100, (getEffectiveBufferSize() * 100) / requiredSampleLength)
        binding.progressBar.progress = collectionProgress
        binding.progressTextView.text = getString(R.string.progress_text, collectionProgress)
    }

    private fun predictBloodPressure() {
        // 检查是否有足够的数据
        if (getEffectiveBufferSize() < requiredSampleLength) {
            Log.w(TAG, "Data insufficient: ${getEffectiveBufferSize()} < $requiredSampleLength")
            return
        }

        try {
            // 获取有效的缓冲区数据
            val effectiveData = getEffectiveBufferData()
            Log.d(TAG, "Effective data length: ${effectiveData.size}")

            // 确保信号处理过程中的任何错误都不会导致应用崩溃
            val features = extractFeatures(effectiveData)
            Log.d(TAG, "Feature extraction complete: number of features=${features.size}")

            val (systolic, diastolic) = bpPredictor.predict(features)
            Log.d(TAG, "Model prediction result: Systolic=$systolic, Diastolic=$diastolic")

            // 检查预测结果是否有效（非零值）
            if (systolic <= 0 || diastolic <= 0) {
                Log.w(TAG, "Invalid prediction result: Systolic=$systolic, Diastolic=$diastolic")
                // 不更新UI，继续收集数据
                return
            }

            runOnUiThread {
                binding.systolicTextView.text = getString(R.string.systolic_pressure, systolic.toInt())
                binding.diastolicTextView.text = getString(R.string.diastolic_pressure, diastolic.toInt())
                Log.d(TAG, "UI updated: Systolic=${systolic.toInt()}, Diastolic=${diastolic.toInt()}")
                
                // 只有在获得有效结果时才锁定结果
                lockResult()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error during blood pressure prediction: ${e.message}", e)
            // 使用默认值以防止应用崩溃
            runOnUiThread {
                binding.systolicTextView.text = getString(R.string.systolic_pressure, 120)
                binding.diastolicTextView.text = getString(R.string.diastolic_pressure, 80)
                Toast.makeText(this, "Blood pressure prediction failed, please try again", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun extractFeatures(signal: List<Float>): FloatArray {
        Log.d(TAG, "Starting feature extraction, signal length: ${signal.size}")

        // 获取特征名称列表，用于验证
        val featureNames = bpPredictor.getFeatureNames()

        // 创建最终特征数组
        val features = FloatArray(bpPredictor.getInputDimension())

        try {
            // 1. 预处理信号 - 确保长度足够
            if (signal.size < requiredSampleLength) {
                Log.w(TAG, "Signal length insufficient, cannot extract features")
                return features
            }

            // 使用适当的窗口提取最新的信号数据
            val processedSignal = signal.takeLast(requiredSampleLength)

            // 2. 计算一阶导数 (VPG) 和二阶导数 (APG)
            val vpg = FeatureExtractor.calculateVPG(processedSignal)
            val apg = FeatureExtractor.calculateAPG(vpg)

            // 3. 计算直方图特征
            val (ppgHistUp, ppgHistDown) = FeatureExtractor.calculateHistogramFeatures(processedSignal, 10)
            val (vpgHistUp, vpgHistDown) = FeatureExtractor.calculateHistogramFeatures(vpg, 10)
            val (apgHistUp, apgHistDown) = FeatureExtractor.calculateHistogramFeatures(apg, 10)

            // 4. 计算邻域最大/最小值特征
            val (ppgMaxNeighborMean, ppgMinNeighborMean) = FeatureExtractor.calculateNeighborExtremumFeatures(processedSignal)
            val (vpgMaxNeighborMean, vpgMinNeighborMean) = FeatureExtractor.calculateNeighborExtremumFeatures(vpg)
            val (apgMaxNeighborMean, apgMinNeighborMean) = FeatureExtractor.calculateNeighborExtremumFeatures(apg)

            // 5. 检测APG特征点
            val apgPoints = FeatureExtractor.detectAPGPoints(processedSignal, vpg, apg)

            // 6. 提取时间域特征
            val timeFeatures = FeatureExtractor.extractTimeFeatures(processedSignal)

            // 7. 计算其他统计特征
            val ppgMean = processedSignal.average().toFloat()
            val ppgStd = sqrt(processedSignal.map { (it - ppgMean) * (it - ppgMean) }.average().toFloat())
            val ppgMax = processedSignal.maxOrNull() ?: 0f
            val ppgMin = processedSignal.minOrNull() ?: 0f

            // 计算信号质量指标
            val skewness = calculateSkewness(processedSignal, ppgMean, ppgStd)
            val kurtosis = calculateKurtosis(processedSignal, ppgMean, ppgStd)

            // 从CSV特征列表中填充特征数组
            // 创建特征映射表
            val featureMap = mutableMapOf<String, Float>()

            // 添加PPG/VPG/APG统计特征
            featureMap["ppg3_max_0"] = ppgMax
            featureMap["ppg4_max_0"] = ppgMax
            featureMap["vpg_min_0"] = vpgMinNeighborMean
            featureMap["apg_max_0"] = apgMaxNeighborMean
            featureMap["ppg_min_0"] = ppgMin
            featureMap["vpg_max_0"] = vpgMaxNeighborMean
            featureMap["apg_min_0"] = apgMinNeighborMean

            // 添加直方图特征
            for (i in 0 until min(ppgHistDown.size, 10)) {
                featureMap["ppg_histogram_down_$i"] = ppgHistDown[i]
                featureMap["ppg3_histogram_down_$i"] = ppgHistDown[i]
                featureMap["ppg4_histogram_down_$i"] = ppgHistDown[i]
            }

            for (i in 0 until min(ppgHistUp.size, 5)) {
                featureMap["ppg_histogram_up_$i"] = ppgHistUp[i]
                featureMap["ppg3_histogram_up_$i"] = ppgHistUp[i]
                featureMap["ppg4_histogram_up_$i"] = ppgHistUp[i]
            }

            for (i in 0 until min(vpgHistDown.size, 5)) {
                featureMap["vpg_histogram_down_$i"] = vpgHistDown[i]
            }

            for (i in 0 until min(vpgHistUp.size, 5)) {
                featureMap["vpg_histogram_up_$i"] = vpgHistUp[i]
            }

            for (i in 0 until min(apgHistDown.size, 8)) {
                featureMap["apg_histogram_down_$i"] = apgHistDown[i]
            }

            for (i in 0 until min(apgHistUp.size, 8)) {
                featureMap["apg_histogram_up_$i"] = apgHistUp[i]
            }

            // 添加邻域特征
            featureMap["ppg_max_neighbor_mean_0"] = ppgMaxNeighborMean
            featureMap["ppg3_max_neighbor_mean_0"] = ppgMaxNeighborMean
            featureMap["ppg4_max_neighbor_mean_0"] = ppgMaxNeighborMean
            featureMap["ppg_min_neighbor_mean_0"] = ppgMinNeighborMean
            featureMap["apg_max_neighbor_mean_0"] = apgMaxNeighborMean
            featureMap["vpg_max_neighbor_mean_0"] = vpgMaxNeighborMean

            // 添加DSDC/USDC特征 (区别信号相关性特征)
            // 使用与DL项目一致的计算方法
            val dsdcFeatures = calculateDSDCFeatures(processedSignal, vpg, apg)
            for ((key, value) in dsdcFeatures) {
                featureMap[key] = value
            }

            // 添加usdc特征
            val usdcFeatures = calculateUSDCFeatures(processedSignal, vpg, apg)
            for ((key, value) in usdcFeatures) {
                featureMap[key] = value
            }

            // 添加循环匹配特征
            featureMap["ppg_mean_cycles_match_peak_59"] = ppgMean

            // 添加APG点特征和时间特征
            for ((key, value) in apgPoints) {
                featureMap[key] = value
            }

            // 添加时间域特征
            for ((key, value) in timeFeatures) {
                featureMap[key] = value
            }

            // 计算波形区域特征
            val sw25 = timeFeatures["Td"] ?: 0f
            val dw50 = timeFeatures["TdiaRise"] ?: 0f
            val dw75 = timeFeatures["TdiaToEnd"] ?: 0f

            featureMap["SW25"] = sw25
            featureMap["SW50"] = sw25 * 2
            featureMap["SW75"] = sw25 * 3
            featureMap["DW50"] = dw50
            featureMap["DW75_norm"] = dw75 / (sw25 * 3).coerceAtLeast(0.1f)  // 防止除零
            featureMap["DW50_norm"] = dw50 / (sw25 * 2).coerceAtLeast(0.1f)  // 防止除零
            featureMap["DWdivSW25"] = dw50 / sw25.coerceAtLeast(0.1f)  // 防止除零
            featureMap["DWdivSW50"] = dw50 / (sw25 * 2).coerceAtLeast(0.1f)  // 防止除零
            featureMap["DWdivSW75"] = dw75 / (sw25 * 3).coerceAtLeast(0.1f)  // 防止除零
            featureMap["SWaddDW75"] = (sw25 * 3) + dw75

            // 计算质量指标
            featureMap["SQI_skew"] = skewness
            featureMap["SQI_kurtosis"] = kurtosis

            // 添加bcda和sdoo特征
            featureMap["bcda"] = (apgPoints["apg_b"] ?: 0f) + (apgPoints["apg_c"] ?: 0f) +
                    (apgPoints["apg_d"] ?: 0f) + (apgPoints["apg_a"] ?: 0f)
            featureMap["sdoo"] = (apgPoints["T_d"] ?: 0f) * (apgPoints["apg_d"] ?: 0f)

            // 计算S3, S4等特征
            featureMap["S3_norm"] = timeFeatures["Ts_norm"] ?: 0f
            featureMap["S4"] = timeFeatures["TNegSteepest"] ?: 0f

            // 计算Ratio特征
            featureMap["Ratio"] = timeFeatures["TSystoDiaRise"] ?: 0f

            // 定义T_peak相关特征
            featureMap["T_peak_a"] = apgPoints["T_a"] ?: 0f
            featureMap["T_peak_b"] = apgPoints["T_b"] ?: 0f
            featureMap["T_peak_c"] = apgPoints["T_c"] ?: 0f
            featureMap["T_peak_d"] = apgPoints["T_d"] ?: 0f
            featureMap["T_peak_e"] = apgPoints["T_e"] ?: 0f

            featureMap["T_peak_a_norm"] = apgPoints["T_a_norm"] ?: 0f
            featureMap["T_peak_b_norm"] = apgPoints["T_b_norm"] ?: 0f
            featureMap["T_peak_c_norm"] = apgPoints["T_c_norm"] ?: 0f
            featureMap["T_peak_d_norm"] = apgPoints["T_d_norm"] ?: 0f
            featureMap["T_peak_e_norm"] = apgPoints["T_e_norm"] ?: 0f

            // 定义AI特征
            featureMap["AI"] = apgPoints["AI"] ?: 0f

            // 添加Tc特征（与T_c相同）
            featureMap["Tc"] = apgPoints["T_c"] ?: 0f

            // 从featureMap填充特征数组
            // 使用特征名称列表确保顺序一致
            if (featureNames != null && featureNames.size == features.size) {
                for (i in featureNames.indices) {
                    val featureName = featureNames[i]
                    features[i] = featureMap[featureName] ?: 0f
                }
                Log.d(TAG, "Filled feature array using predefined feature order")
            } else {
                // 如果没有特征名称列表，使用固定顺序
                val featureOrder = listOf(
                    "ppg3_histogram_down_8", "TNegSteepest_norm", "AUCsys_norm", "ppg_histogram_down_2",
                    "T_peak_e", "S3_norm", "T_peak_c_norm", "T_peak_b_norm", "ppg_mean_cycles_match_peak_59",
                    "dsdc_12", "dsdc_11", "dsdc_13", "dsdc_14", "dsdc_10", "ppg3_max_neighbor_mean_0",
                    "Ts_norm", "SW75", "T_peak_a_norm", "T_peak_d_norm", "vpg_histogram_down_3",
                    "sdoo", "dsdc_16", "apg_histogram_down_5", "T_peak_d", "TNegSteepest",
                    "TdiaRise_norm", "S4", "apg_histogram_down_3", "SQI_kurtosis", "dsdc_15",
                    "apg_b", "SW50", "apg_histogram_down_6", "dsdc_1", "SteepDiaRise",
                    "dsdc_9", "T_peak_e_norm", "AUCsys", "apg_a", "vpg_max_neighbor_mean_0",
                    "DWdivSW25", "usdc_5", "ppg4_histogram_down_4", "ppg3_histogram_down_9", "ppg4_histogram_down_6",
                    "vpg_histogram_up_1", "DWdivSW50", "TdiaRise", "ppg4_histogram_up_4", "DWdivSW75",
                    "dsdc_17", "ppg4_histogram_down_2", "vpg_histogram_down_0", "apg_histogram_down_4", "ppg3_max_0",
                    "dsdc_2", "TdiaToEnd", "T_peak_a", "vpg_histogram_down_2", "T_c",
                    "vpg_min_0", "apg_histogram_down_7", "T_c_norm", "Td", "ppg3_histogram_down_4",
                    "ppg_histogram_down_4", "ppg3_histogram_down_3", "dsdc_8", "SW25", "SQI_skew",
                    "ppg4_max_0", "TSystoDiaRise", "apg_max_neighbor_mean_0", "Ratio", "ppg_histogram_down_5",
                    "dsdc_5", "SWaddDW75", "dsdc_4", "ppg_histogram_down_3", "Tc",
                    "T_e_norm", "apg_c", "ppg3_histogram_down_5", "T_d", "T_d_norm",
                    "dsdc_6", "ppg_histogram_down_1", "ppg3_histogram_up_4", "ppg4_histogram_down_3", "apg_e",
                    "apg_max_0", "DW75_norm", "apg_histogram_down_0", "DW50", "ppg3_histogram_up_3",
                    "apg_d", "DW50_norm", "vpg_histogram_up_0", "bcda", "AI"
                )

                // 填充特征数组，如果某个特征不存在则使用0
                var missingFeatures = 0
                for (i in 0 until min(featureOrder.size, features.size)) {
                    val feature = featureMap[featureOrder[i]]
                    if (feature == null) {
                        missingFeatures++
                        features[i] = 0f
                    } else {
                        features[i] = feature
                    }
                }

                if (missingFeatures > 0) {
                    Log.w(TAG, "Missing $missingFeatures features, filled with 0")
                }
            }

            Log.d(TAG, "Feature extraction complete, total $features.size features")

        } catch (e: Exception) {
            Log.e(TAG, "Feature extraction failed: ${e.message}", e)
            // 确保返回零填充的特征数组，不会导致应用崩溃
            for (i in features.indices) {
                features[i] = 0f
            }
        }

        return features
    }

    // 计算偏度
    private fun calculateSkewness(signal: List<Float>, mean: Float, stdDev: Float): Float {
        if (signal.size < 3 || stdDev == 0f) return 0f

        var sum = 0f
        for (value in signal) {
            sum += ((value - mean) / stdDev).pow(3)
        }

        return sum / (signal.size - 1)
    }

    // 计算峰度
    private fun calculateKurtosis(signal: List<Float>, mean: Float, stdDev: Float): Float {
        if (signal.size < 4 || stdDev == 0f) return 0f

        var sum = 0f
        for (value in signal) {
            sum += ((value - mean) / stdDev).pow(4)
        }

        return sum / (signal.size - 1) - 3f  // 减去3使正态分布的峰度为0
    }

    // 计算DSDC特征 - 与DL项目保持一致
    private fun calculateDSDCFeatures(ppg: List<Float>, vpg: List<Float>, apg: List<Float>): Map<String, Float> {
        val result = mutableMapOf<String, Float>()

        // 计算PPG信号的统计特征
        val ppgMean = ppg.average().toFloat()
        val ppgStd = sqrt(ppg.map { (it - ppgMean) * (it - ppgMean) }.average().toFloat())
        val ppgMax = ppg.maxOrNull() ?: 0f
        val ppgMin = ppg.minOrNull() ?: 0f

        // 计算VPG信号的统计特征
        val vpgMean = vpg.average().toFloat()
        val vpgStd = sqrt(vpg.map { (it - vpgMean) * (it - vpgMean) }.average().toFloat())
        val vpgMax = vpg.maxOrNull() ?: 0f
        val vpgMin = vpg.minOrNull() ?: 0f

        // 计算APG信号的统计特征
        val apgMean = apg.average().toFloat()
        val apgStd = sqrt(apg.map { (it - apgMean) * (it - apgMean) }.average().toFloat())
        val apgMax = apg.maxOrNull() ?: 0f
        val apgMin = apg.minOrNull() ?: 0f

        // 填充DSDC特征
        result["dsdc_0"] = ppgMean
        result["dsdc_1"] = ppgStd
        result["dsdc_2"] = ppgMax
        result["dsdc_3"] = ppgMin
        result["dsdc_4"] = ppgMax - ppgMin
        result["dsdc_5"] = vpgMean
        result["dsdc_6"] = vpgStd
        result["dsdc_7"] = vpgMax
        result["dsdc_8"] = vpgMin
        result["dsdc_9"] = vpgMax - vpgMin
        result["dsdc_10"] = apgMean
        result["dsdc_11"] = apgStd
        result["dsdc_12"] = apgMax
        result["dsdc_13"] = apgMin
        result["dsdc_14"] = apgMax - apgMin

        // 计算信号之间的相关性
        val ppgVpgCorr = calculateCorrelation(ppg.take(vpg.size), vpg)
        val ppgApgCorr = calculateCorrelation(ppg.take(apg.size), apg)
        val vpgApgCorr = calculateCorrelation(vpg.take(apg.size), apg)

        result["dsdc_15"] = ppgVpgCorr
        result["dsdc_16"] = ppgApgCorr
        result["dsdc_17"] = vpgApgCorr

        return result
    }

    // 计算USDC特征 - 与DL项目保持一致
    private fun calculateUSDCFeatures(ppg: List<Float>, vpg: List<Float>, apg: List<Float>): Map<String, Float> {
        val result = mutableMapOf<String, Float>()

        // 使用与DL项目一致的计算方法
        val (ppgMaxNeighborMean, ppgMinNeighborMean) = FeatureExtractor.calculateNeighborExtremumFeatures(ppg)
        val (vpgMaxNeighborMean, vpgMinNeighborMean) = FeatureExtractor.calculateNeighborExtremumFeatures(vpg)
        val (apgMaxNeighborMean, apgMinNeighborMean) = FeatureExtractor.calculateNeighborExtremumFeatures(apg)

        result["usdc_0"] = ppgMaxNeighborMean
        result["usdc_1"] = ppgMinNeighborMean
        result["usdc_2"] = vpgMaxNeighborMean
        result["usdc_3"] = vpgMinNeighborMean
        result["usdc_4"] = apgMaxNeighborMean
        result["usdc_5"] = apgMinNeighborMean

        return result
    }

    // 计算两个信号之间的相关性
    private fun calculateCorrelation(signal1: List<Float>, signal2: List<Float>): Float {
        if (signal1.size != signal2.size || signal1.isEmpty()) return 0f

        val mean1 = signal1.average().toFloat()
        val mean2 = signal2.average().toFloat()

        var sum12 = 0f
        var sum1Sq = 0f
        var sum2Sq = 0f

        for (i in signal1.indices) {
            val diff1 = signal1[i] - mean1
            val diff2 = signal2[i] - mean2

            sum12 += diff1 * diff2
            sum1Sq += diff1 * diff1
            sum2Sq += diff2 * diff2
        }

        val denominator = sqrt(sum1Sq * sum2Sq)
        return if (denominator > 0) sum12 / denominator else 0f
    }

    private fun lockResult() {
        isResultLocked = true

        try {
            val systolicText = binding.systolicTextView.text.toString()
            val diastolicText = binding.diastolicTextView.text.toString()

            val systolicMatch = Regex("\\d+").find(systolicText)
            val diastolicMatch = Regex("\\d+").find(diastolicText)

            lockedSystolic = systolicMatch?.value?.toInt() ?: 0
            lockedDiastolic = diastolicMatch?.value?.toInt() ?: 0

            Log.d(TAG, "Blood pressure values locked: Systolic=$lockedSystolic, Diastolic=$lockedDiastolic")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to extract blood pressure values", e)
            lockedSystolic = 0
            lockedDiastolic = 0
        }

        stopMonitoring()

        binding.systolicTextView.text = getString(R.string.systolic_pressure, lockedSystolic)
        binding.diastolicTextView.text = getString(R.string.diastolic_pressure, lockedDiastolic)
        binding.cameraStatusTextView.text = "Measurement Complete"
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                setupCamera()
            } else {
                Toast.makeText(
                    this,
                    getString(R.string.camera_permission_denied),
                    Toast.LENGTH_SHORT
                ).show()
                finish()
            }
        }
    }

    // 添加生命周期管理方法
    override fun onPause() {
        super.onPause()
        // 暂停时停止监测并释放相机资源
        if (isMonitoring) {
            stopMonitoring()
        }
        releaseCamera()
    }

    override fun onResume() {
        super.onResume()
        // 恢复时重新设置相机
        if (allPermissionsGranted() && camera == null) {
            setupCamera()
        }
    }

    // 释放相机资源
    private fun releaseCamera() {
        try {
            camera?.cameraControl?.enableTorch(false)
            camera = null
            cameraProvider?.unbindAll()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to release camera resources", e)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        // 确保在Activity销毁时释放所有资源
        releaseCamera()
        cameraExecutor.shutdown()
        try {
            bpPredictor.close()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to close blood pressure predictor", e)
        }
    }

    // 获取有效的缓冲区大小
    private fun getEffectiveBufferSize(): Int {
        return if (ppgSignalBufferFull) ppgWindowSize else ppgSignalBufferIndex
    }
    
    // 获取缓冲区中的有效数据作为列表
    private fun getEffectiveBufferData(): List<Float> {
        val result = ArrayList<Float>(requiredSampleLength)
        
        if (ppgSignalBufferFull) {
            // 缓冲区已满，获取最近的requiredSampleLength个样本
            val startIdx = (ppgSignalBufferIndex + ppgWindowSize - requiredSampleLength) % ppgWindowSize
            for (i in 0 until requiredSampleLength) {
                val idx = (startIdx + i) % ppgWindowSize
                result.add(ppgSignalBuffer[idx])
            }
        } else if (ppgSignalBufferIndex >= requiredSampleLength) {
            // 缓冲区未满但数据足够
            for (i in ppgSignalBufferIndex - requiredSampleLength until ppgSignalBufferIndex) {
                result.add(ppgSignalBuffer[i])
            }
        }
        
        return result
    }

    companion object {
        private const val TAG = "PPGBloodPressureApp"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}

private class PpgAnalyzer(private val onSignalReady: (Float) -> Unit) : ImageAnalysis.Analyzer {
    
    override fun analyze(image: ImageProxy) {
        val redMean = extractRedMean(image)
        
        onSignalReady(redMean)
        
        image.close()
    }
    
    private fun extractRedMean(image: ImageProxy): Float {
        val buffer = image.planes[0].buffer
        val data = ByteArray(buffer.remaining())
        buffer.get(data)
        
        var totalRed = 0L
        var pixelCount = 0
        
        val width = image.width
        val height = image.height
        
        val centerStartX = width * 4 / 10
        val centerEndX = width * 6 / 10
        val centerStartY = height * 4 / 10
        val centerEndY = height * 6 / 10
        
        val rowStride = image.planes[0].rowStride
        val pixelStride = image.planes[0].pixelStride
        
        for (y in centerStartY until centerEndY) {
            for (x in centerStartX until centerEndX) {
                val index = y * rowStride + x * pixelStride
                if (index < data.size) {
                    val pixelValue = data[index].toInt() and 0xFF
                    totalRed += pixelValue.toLong()
                    pixelCount++
                }
            }
        }
        
        return if (pixelCount > 0) totalRed.toFloat() / pixelCount else 0f
    }
} 