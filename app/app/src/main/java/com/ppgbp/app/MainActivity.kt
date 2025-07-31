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


class FeatureExtractor {
    companion object {
        //  Calculating the First Derivative (VPG) from the PPG Signal
        fun calculateVPG(ppg: List<Float>): List<Float> {
            val vpg = mutableListOf<Float>()
            for (i in 1 until ppg.size) {
                vpg.add(ppg[i] - ppg[i-1])
            }
            return vpg
        }

        // Compute the second derivative (APG) from the VPG signal.
        fun calculateAPG(vpg: List<Float>): List<Float> {
            val apg = mutableListOf<Float>()
            for (i in 1 until vpg.size) {
                apg.add(vpg[i] - vpg[i-1])
            }
            return apg
        }

        // Compute the histogram features of the signal.
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

            // Normalize the histogram.
            val totalUp = upHistogram.sum()
            val totalDown = downHistogram.sum()

            val normalizedUp = if (totalUp > 0) upHistogram.map { it / totalUp } else upHistogram
            val normalizedDown = if (totalDown > 0) downHistogram.map { it / totalDown } else downHistogram

            return Pair(normalizedUp, normalizedDown)
        }

        // Compute the neighborhood maximum/minimum features.
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

        // Detect the characteristic points (a, b, c, d, e) in the PPG waveform.
        fun detectAPGPoints(ppg: List<Float>, vpg: List<Float>, apg: List<Float>): Map<String, Float> {
            val result = mutableMapOf<String, Float>()

            // Find the start point of systole.
            val peaks = mutableListOf<Int>()
            for (i in 2 until ppg.size - 2) {
                if (ppg[i] > ppg[i - 1] && ppg[i] > ppg[i + 1] &&
                    ppg[i] > ppg[i - 2] && ppg[i] > ppg[i + 2]) {
                    peaks.add(i)
                }
            }

            if (peaks.isEmpty()) return result

            // Compute the characteristic points for each cardiac cycle.
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

                // Point a is the positive peak of the VPG signal.
                aPoint = vpg[aIndex]

                // Find point b (the first negative extremum) after point a.
                for (i in aIndex + 1 until min(aIndex + 20, vpg.size)) {
                    if (i < vpg.size - 1 &&
                        vpg[i] < vpg[i - 1] && vpg[i] < vpg[i + 1]) {
                        bIndex = i
                        bPoint = vpg[i]
                        break
                    }
                }

                // Find point c (the second positive extremum) after point b.
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

                // Find point d (the second negative extremum) after point c.
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

                // Find point e (the third positive extremum) after point d.
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

                // Save the point values.
                points.getOrPut("a") { mutableListOf() }.add(aPoint)
                points.getOrPut("b") { mutableListOf() }.add(bPoint)
                points.getOrPut("c") { mutableListOf() }.add(cPoint)
                points.getOrPut("d") { mutableListOf() }.add(dPoint)
                points.getOrPut("e") { mutableListOf() }.add(ePoint)

                // Save the time points (relative distance from point a).
                if (bIndex > 0) timings.getOrPut("b") { mutableListOf() }.add((bIndex - aIndex).toFloat())
                if (cIndex > 0) timings.getOrPut("c") { mutableListOf() }.add((cIndex - aIndex).toFloat())
                if (dIndex > 0) timings.getOrPut("d") { mutableListOf() }.add((dIndex - aIndex).toFloat())
                if (eIndex > 0) timings.getOrPut("e") { mutableListOf() }.add((eIndex - aIndex).toFloat())
            }

            // Compute the average value.
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

            // Compute the ratio features.
            if (apgA != 0f) {
                result["ratio_apg_b"] = apgB / apgA
                result["ratio_apg_c"] = apgC / apgA
                result["ratio_apg_d"] = apgD / apgA
                result["ratio_apg_e"] = apgE / apgA
            }

            // Compute the time-based features.
            val tB = timings["b"]?.average()?.toFloat() ?: 0f
            val tC = timings["c"]?.average()?.toFloat() ?: 0f
            val tD = timings["d"]?.average()?.toFloat() ?: 0f
            val tE = timings["e"]?.average()?.toFloat() ?: 0f

            result["T_b"] = tB
            result["T_c"] = tC
            result["T_d"] = tD
            result["T_e"] = tE

            // Compute the average length of cardiac cycles.
            val cycleDuration = if (peaks.size >= 2) {
                var sum = 0
                for (i in 1 until peaks.size) {
                    sum += peaks[i] - peaks[i-1]
                }
                sum.toFloat() / (peaks.size - 1)
            } else {
                0f
            }

            // Normalize the time-based features.
            if (cycleDuration > 0) {
                result["T_a_norm"] = 0f
                result["T_b_norm"] = tB / cycleDuration
                result["T_c_norm"] = tC / cycleDuration
                result["T_d_norm"] = tD / cycleDuration
                result["T_e_norm"] = tE / cycleDuration
            }

            // Compute other features.
            val peakValues = peaks.map { ppg[it] }
            val avgPeakValue = peakValues.average().toFloat()

            // Compute the relative position features of waveform peaks.
            for (peakIndex in peaks) {
                if (peakIndex >= ppg.size - 1) continue

                // Find the valley points after the peaks.
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

                    // Compute the areas of systole and diastole.
                    val systolicArea = (0 until valleyIndex - peakIndex).sumOf {
                        (ppg[peakIndex + it] - ppg[valleyIndex]).toDouble()
                    }.toFloat()

                    result["AUCsys"] = (result["AUCsys"] ?: 0f) + systolicArea

                    if (avgPeakValue > 0) {
                        result["AUCsys_norm"] = (result["AUCsys_norm"] ?: 0f) + (systolicArea / avgPeakValue)
                    }
                }
            }

            // Compute the average values for processing.
            val peakCount = peaks.size.toFloat()
            if (peakCount > 0) {
                result["Ts"] = (result["Ts"] ?: 0f) / peakCount
                result["Ts_norm"] = (result["Ts_norm"] ?: 0f) / peakCount
                result["AUCsys"] = (result["AUCsys"] ?: 0f) / peakCount
                result["AUCsys_norm"] = (result["AUCsys_norm"] ?: 0f) / peakCount
            }

            // Compute the Augmentation Index (AI) as a vascular elasticity measure.
            if (result["apg_d"] != null && result["apg_a"] != null && result["apg_a"] != 0f) {
                result["AI"] = result["apg_d"]!! / result["apg_a"]!!
            }

            return result
        }

        // Extract DSP (Digital Signal Processing) features.
        fun extractTimeFeatures(ppg: List<Float>): Map<String, Float> {
            val result = mutableMapOf<String, Float>()

            // Detect peaks and valleys in the signal.
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

            // Compute systolic time, diastolic time, and total time.
            var totalSysTime = 0f
            var totalDiaTime = 0f
            var totalSteepest = 0f
            var totalNegSteepest = 0f
            var count = 0

            for (peakIndex in peaks) {
                // Find the valley before this peak.
                var prevValley = -1
                for (v in valleys) {
                    if (v < peakIndex) prevValley = v
                    else break
                }

                // Find the valley after this peak.
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

                    // Compute the steepest rising and falling slopes.
                    var maxSlope = Float.MIN_VALUE
                    var minSlope = Float.MAX_VALUE

                    // Compute the slope of the ascending segment.
                    for (i in prevValley until peakIndex) {
                        val slope = ppg[i+1] - ppg[i]
                        if (slope > maxSlope) maxSlope = slope
                    }

                    // Compute the slope of the descending segment.
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

                // Normalize the features.
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


    private val ppgSignalBuffer = FloatArray(1800)
    private var ppgSignalBufferIndex = 0
    private var ppgSignalBufferFull = false

    private val displaySignalBuffer = mutableListOf<Entry>()
    private val ppgWindowSize = 1800
    private val displayWindowSize = 200
    private val requiredSampleLength = 450

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


        Log.i(TAG, "Application initialization completed")
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

        val emptyDataSet = LineDataSet(ArrayList(), "PPG信号")
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


                        ppgSignalBufferIndex = 0
                        ppgSignalBufferFull = false
                        displaySignalBuffer.clear()

                        currentMeanValue = redMean
                        updateYAxisRange()
                        

                        Log.d(TAG, " Signal is stable, starting data collection")
                    }
                } else {
                    signalStabilityCounter = 0
                }
                lastMean = redMean
                binding.cameraStatusTextView.text = "Waiting for signal stability..."
                return
            }

            binding.cameraStatusTextView.text = getString(R.string.measuring_text)


            // Use circular buffer mode to write data
            ppgSignalBuffer[ppgSignalBufferIndex] = redMean
            ppgSignalBufferIndex = (ppgSignalBufferIndex + 1) % ppgWindowSize
            if (ppgSignalBufferIndex == 0) {
                ppgSignalBufferFull = true

                Log.d(TAG, "The buffer is filled for one round ")
            }

            currentMeanValue = currentMeanValue * (1 - smoothingFactor) + redMean * smoothingFactor
            updateYAxisRange()

            updateSignalChart()

            updateProgress()

            if (getEffectiveBufferSize() >= requiredSampleLength && collectionProgress >= 100) {
                Log.d(TAG, "Data collection completed, starting blood pressure prediction")
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

        displaySignalBuffer.clear()


        val effectiveSize = getEffectiveBufferSize()
        val displayCount = min(effectiveSize, displayWindowSize)

        if (displayCount > 0) {
            if (ppgSignalBufferFull) {

                val startIdx = (ppgSignalBufferIndex + ppgWindowSize - displayCount) % ppgWindowSize
                for (i in 0 until displayCount) {
                    val idx = (startIdx + i) % ppgWindowSize
                    val x = i.toFloat()
                    displaySignalBuffer.add(Entry(x, ppgSignalBuffer[idx]))
                }
            } else {

                val startIdx = max(0, ppgSignalBufferIndex - displayCount)
                for (i in 0 until displayCount) {
                    val idx = startIdx + i
                    val x = i.toFloat()
                    displaySignalBuffer.add(Entry(x, ppgSignalBuffer[idx]))
                }
            }
        }

        val dataSet = LineDataSet(displaySignalBuffer, "PPG Signal")
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
        // Check if there is enough data
        if (getEffectiveBufferSize() < requiredSampleLength) {
            Log.w(TAG, "Insufficient data: ${getEffectiveBufferSize()} < $requiredSampleLength")
            return
        }

        try {
            // Get valid buffer data
            val effectiveData = getEffectiveBufferData()
            Log.d(TAG, "Effective data length: ${effectiveData.size}")

            // Ensure that any errors during signal processing do not cause the app to crash
            val features = extractFeatures(effectiveData)
            Log.d(TAG, "Feature extraction completed: Number of features=${features.size}")

            val (systolic, diastolic) = bpPredictor.predict(features)
            Log.d(TAG, "Model prediction result: Systolic=$systolic, Diastolic=$diastolic")

            // Check if the prediction result is valid (non-zero values)
            if (systolic <= 0 || diastolic <= 0) {
                Log.w(TAG, "Invalid prediction result: Systolic=$systolic, Diastolic=$diastolic")
                // Do not update UI, continue collecting data
                return
            }

            runOnUiThread {
                binding.systolicTextView.text = getString(R.string.systolic_pressure, systolic.toInt())
                binding.diastolicTextView.text = getString(R.string.diastolic_pressure, diastolic.toInt())
                Log.d(TAG, "UI update completed: Systolic=${systolic.toInt()}, Diastolic=${diastolic.toInt()}")


                // Only lock the result when a valid result is obtained
                lockResult()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error occurred during blood pressure prediction: ${e.message}", e)

            // Use default values to prevent the app from crashing
            runOnUiThread {
                binding.systolicTextView.text = getString(R.string.systolic_pressure, 120)
                binding.diastolicTextView.text = getString(R.string.diastolic_pressure, 80)
                Toast.makeText(this, "Blood pressure prediction failed, please try again", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun extractFeatures(signal: List<Float>): FloatArray {
        Log.d(TAG, "Starting feature extraction, signal length: ${signal.size}")


        // Get the list of feature names for validation
        val featureNames = bpPredictor.getFeatureNames()


        // Create the final feature array
        val features = FloatArray(bpPredictor.getInputDimension())

        try {
            // 1. Preprocess the signal - Ensure sufficient length
            if (signal.size < requiredSampleLength) {
                Log.w(TAG, "Signal length is insufficient, cannot extract features")
                return features
            }

            // Use an appropriate window to extract the latest signal data
            val processedSignal = signal.takeLast(requiredSampleLength)

            // 2. Calculate the first derivative (VPG) and second derivative (APG)
            val vpg = FeatureExtractor.calculateVPG(processedSignal)
            val apg = FeatureExtractor.calculateAPG(vpg)

            // 3. Calculate histogram features
            val (ppgHistUp, ppgHistDown) = FeatureExtractor.calculateHistogramFeatures(processedSignal, 10)
            val (vpgHistUp, vpgHistDown) = FeatureExtractor.calculateHistogramFeatures(vpg, 10)
            val (apgHistUp, apgHistDown) = FeatureExtractor.calculateHistogramFeatures(apg, 10)

            // 4. Calculate neighborhood maximum/minimum features
            val (ppgMaxNeighborMean, ppgMinNeighborMean) = FeatureExtractor.calculateNeighborExtremumFeatures(processedSignal)
            val (vpgMaxNeighborMean, vpgMinNeighborMean) = FeatureExtractor.calculateNeighborExtremumFeatures(vpg)
            val (apgMaxNeighborMean, apgMinNeighborMean) = FeatureExtractor.calculateNeighborExtremumFeatures(apg)

            // 5. Detect APG feature points
            val apgPoints = FeatureExtractor.detectAPGPoints(processedSignal, vpg, apg)

            // 6. Extract time-domain features
            val timeFeatures = FeatureExtractor.extractTimeFeatures(processedSignal)

            // 7. Calculate additional statistical features
            val ppgMean = processedSignal.average().toFloat()
            val ppgStd = sqrt(processedSignal.map { (it - ppgMean) * (it - ppgMean) }.average().toFloat())
            val ppgMax = processedSignal.maxOrNull() ?: 0f
            val ppgMin = processedSignal.minOrNull() ?: 0f

            // Calculate signal quality metrics
            val skewness = calculateSkewness(processedSignal, ppgMean, ppgStd)
            val kurtosis = calculateKurtosis(processedSignal, ppgMean, ppgStd)

            // Fill the feature array from the CSV feature list
            // Create a feature mapping table
            val featureMap = mutableMapOf<String, Float>()


            // Add PPG/VPG/APG statistical features
            featureMap["ppg3_max_0"] = ppgMax
            featureMap["ppg4_max_0"] = ppgMax
            featureMap["vpg_min_0"] = vpgMinNeighborMean
            featureMap["apg_max_0"] = apgMaxNeighborMean
            featureMap["ppg_min_0"] = ppgMin
            featureMap["vpg_max_0"] = vpgMaxNeighborMean
            featureMap["apg_min_0"] = apgMinNeighborMean


            // Add histogram features
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

            // Add neighborhood features
            featureMap["ppg_max_neighbor_mean_0"] = ppgMaxNeighborMean
            featureMap["ppg3_max_neighbor_mean_0"] = ppgMaxNeighborMean
            featureMap["ppg4_max_neighbor_mean_0"] = ppgMaxNeighborMean
            featureMap["ppg_min_neighbor_mean_0"] = ppgMinNeighborMean
            featureMap["apg_max_neighbor_mean_0"] = apgMaxNeighborMean
            featureMap["vpg_max_neighbor_mean_0"] = vpgMaxNeighborMean


            // Add DSDC/USDC features (Differential Signal Dependency Correlation features)

            val dsdcFeatures = calculateDSDCFeatures(processedSignal, vpg, apg)
            for ((key, value) in dsdcFeatures) {
                featureMap[key] = value
            }

            // Add USDC features
            val usdcFeatures = calculateUSDCFeatures(processedSignal, vpg, apg)
            for ((key, value) in usdcFeatures) {
                featureMap[key] = value
            }


            // Add cyclic matching features
            featureMap["ppg_mean_cycles_match_peak_59"] = ppgMean


            // Add APG point features and time-domain features
            for ((key, value) in apgPoints) {
                featureMap[key] = value
            }

            // Add time-domain features
            for ((key, value) in timeFeatures) {
                featureMap[key] = value
            }


            // Calculate waveform region features
            val sw25 = timeFeatures["Td"] ?: 0f
            val dw50 = timeFeatures["TdiaRise"] ?: 0f
            val dw75 = timeFeatures["TdiaToEnd"] ?: 0f

            featureMap["SW25"] = sw25
            featureMap["SW50"] = sw25 * 2
            featureMap["SW75"] = sw25 * 3
            featureMap["DW50"] = dw50
            featureMap["DW75_norm"] = dw75 / (sw25 * 3).coerceAtLeast(0.1f)
            featureMap["DW50_norm"] = dw50 / (sw25 * 2).coerceAtLeast(0.1f)
            featureMap["DWdivSW25"] = dw50 / sw25.coerceAtLeast(0.1f)
            featureMap["DWdivSW50"] = dw50 / (sw25 * 2).coerceAtLeast(0.1f)
            featureMap["DWdivSW75"] = dw75 / (sw25 * 3).coerceAtLeast(0.1f)  // Prevent division by zero
            featureMap["SWaddDW75"] = (sw25 * 3) + dw75


            // Calculate quality metrics
            featureMap["SQI_skew"] = skewness
            featureMap["SQI_kurtosis"] = kurtosis


            // Add BCDA and SDOO features
            featureMap["bcda"] = (apgPoints["apg_b"] ?: 0f) + (apgPoints["apg_c"] ?: 0f) +
                    (apgPoints["apg_d"] ?: 0f) + (apgPoints["apg_a"] ?: 0f)
            featureMap["sdoo"] = (apgPoints["T_d"] ?: 0f) * (apgPoints["apg_d"] ?: 0f)


            // Calculate S3, S4, and other features
            featureMap["S3_norm"] = timeFeatures["Ts_norm"] ?: 0f
            featureMap["S4"] = timeFeatures["TNegSteepest"] ?: 0f

            // Calculate Ratio features
            featureMap["Ratio"] = timeFeatures["TSystoDiaRise"] ?: 0f


            // Define T_peak-related features
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


            // Define AI features
            featureMap["AI"] = apgPoints["AI"] ?: 0f


            // Add Tc feature (same as T_c)
            featureMap["Tc"] = apgPoints["T_c"] ?: 0f


            // Fill the feature array from featureMap
            // Use the feature name list to ensure consistent ordering
            if (featureNames != null && featureNames.size == features.size) {
                for (i in featureNames.indices) {
                    val featureName = featureNames[i]
                    features[i] = featureMap[featureName] ?: 0f
                }
                Log.d(TAG, "Fill the feature array using the predefined feature order")
            } else {

                // If there is no feature name list, use a fixed order
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


                // Fill the feature array, using 0 if a feature is missing
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
                    Log.w(TAG, "Missing ${missingFeatures} features, filled with 0")
                }
            }

            Log.d(TAG, " Feature extraction completed, total of ${features.size} features")

        } catch (e: Exception) {
            Log.e(TAG, "Feature extraction failed: ${e.message}", e)
            for (i in features.indices) {
                features[i] = 0f
            }
        }

        return features
    }

    // Calculate Skewness
    private fun calculateSkewness(signal: List<Float>, mean: Float, stdDev: Float): Float {
        if (signal.size < 3 || stdDev == 0f) return 0f

        var sum = 0f
        for (value in signal) {
            sum += ((value - mean) / stdDev).pow(3)
        }

        return sum / (signal.size - 1)
    }

    // calculate Kurtosis
    private fun calculateKurtosis(signal: List<Float>, mean: Float, stdDev: Float): Float {
        if (signal.size < 4 || stdDev == 0f) return 0f

        var sum = 0f
        for (value in signal) {
            sum += ((value - mean) / stdDev).pow(4)
        }

        return sum / (signal.size - 1) - 3f
    }

    // calculate DSDC Features
    private fun calculateDSDCFeatures(ppg: List<Float>, vpg: List<Float>, apg: List<Float>): Map<String, Float> {
        val result = mutableMapOf<String, Float>()

        // PPG
        val ppgMean = ppg.average().toFloat()
        val ppgStd = sqrt(ppg.map { (it - ppgMean) * (it - ppgMean) }.average().toFloat())
        val ppgMax = ppg.maxOrNull() ?: 0f
        val ppgMin = ppg.minOrNull() ?: 0f

        // VPG
        val vpgMean = vpg.average().toFloat()
        val vpgStd = sqrt(vpg.map { (it - vpgMean) * (it - vpgMean) }.average().toFloat())
        val vpgMax = vpg.maxOrNull() ?: 0f
        val vpgMin = vpg.minOrNull() ?: 0f

        // APG
        val apgMean = apg.average().toFloat()
        val apgStd = sqrt(apg.map { (it - apgMean) * (it - apgMean) }.average().toFloat())
        val apgMax = apg.maxOrNull() ?: 0f
        val apgMin = apg.minOrNull() ?: 0f

        // DSDC
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


        val ppgVpgCorr = calculateCorrelation(ppg.take(vpg.size), vpg)
        val ppgApgCorr = calculateCorrelation(ppg.take(apg.size), apg)
        val vpgApgCorr = calculateCorrelation(vpg.take(apg.size), apg)

        result["dsdc_15"] = ppgVpgCorr
        result["dsdc_16"] = ppgApgCorr
        result["dsdc_17"] = vpgApgCorr

        return result
    }

    // calculate USDC Features
    private fun calculateUSDCFeatures(ppg: List<Float>, vpg: List<Float>, apg: List<Float>): Map<String, Float> {
        val result = mutableMapOf<String, Float>()


    // Use the calculation method consistent with the DL project
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

    // calculate Correlation
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

            Log.d(TAG, "Locked blood pressure values: Systolic=$lockedSystolic,  diastolic =$lockedDiastolic")
        } catch (e: Exception) {
            Log.e(TAG, " Failed to extract blood pressure values", e)
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


    override fun onPause() {
        super.onPause()
        if (isMonitoring) {
            stopMonitoring()
        }
        releaseCamera()
    }

    override fun onResume() {
        super.onResume()
        if (allPermissionsGranted() && camera == null) {
            setupCamera()
        }
    }


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
        releaseCamera()
        cameraExecutor.shutdown()
        try {
            bpPredictor.close()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to close blood pressure predictor", e)
        }
    }


    private fun getEffectiveBufferSize(): Int {
        return if (ppgSignalBufferFull) ppgWindowSize else ppgSignalBufferIndex
    }
    

    private fun getEffectiveBufferData(): List<Float> {
        val result = ArrayList<Float>(requiredSampleLength)
        
        if (ppgSignalBufferFull) {
            val startIdx = (ppgSignalBufferIndex + ppgWindowSize - requiredSampleLength) % ppgWindowSize
            for (i in 0 until requiredSampleLength) {
                val idx = (startIdx + i) % ppgWindowSize
                result.add(ppgSignalBuffer[idx])
            }
        } else if (ppgSignalBufferIndex >= requiredSampleLength) {
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
