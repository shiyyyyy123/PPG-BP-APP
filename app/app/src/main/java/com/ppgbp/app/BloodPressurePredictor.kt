package com.ppgbp.app

import android.content.Context
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import org.json.JSONObject


class BloodPressurePredictor(private val context: Context) {
    
    private val tag = "BloodPressurePredictor"
    

    private val ortEnvironment = OrtEnvironment.getEnvironment()
    

    private var ortSession: OrtSession? = null
    

    private val inputDimension = 100
    

    private var spCorrectorSession: OrtSession? = null
    
    // 添加特征标准化所需的均值和标准差数组
    private var featureMeans: FloatArray? = null
    private var featureStds: FloatArray? = null
    
    // 添加特征名称列表，用于验证
    private var featureNames: List<String>? = null
    
    init {
        try {

            loadModels()
            
            // 加载特征统计信息
            loadFeatureStats()
            
            // 加载特征名称
            loadFeatureNames()
            

            val sessionOptions = OrtSession.SessionOptions()
            sessionOptions.setIntraOpNumThreads(2)
            

            val modelFile = File(context.filesDir, "student_mobile.onnx")
            ortSession = ortEnvironment.createSession(modelFile.absolutePath, sessionOptions)
            

            val spCorrectorFile = File(context.filesDir, "sp_corrector.onnx")
            if (spCorrectorFile.exists()) {
                Log.d(tag, "加载收缩压修正器模型")
                spCorrectorSession = ortEnvironment.createSession(spCorrectorFile.absolutePath, sessionOptions)
            }
            
            Log.d(tag, "模型加载成功")
        } catch (e: Exception) {
            Log.e(tag, "模型加载失败", e)
        }
    }
    
    // 加载特征统计信息（均值和标准差）
    private fun loadFeatureStats() {
        try {
            val statsStream = context.assets.open("feature_stats.json")
            val statsJson = statsStream.bufferedReader().use { it.readText() }
            val statsObject = JSONObject(statsJson)
            
            val meansArray = statsObject.getJSONArray("means")
            val stdsArray = statsObject.getJSONArray("stds")
            
            featureMeans = FloatArray(inputDimension)
            featureStds = FloatArray(inputDimension)
            
            for (i in 0 until min(inputDimension, meansArray.length())) {
                featureMeans!![i] = meansArray.getDouble(i).toFloat()
                featureStds!![i] = stdsArray.getDouble(i).toFloat()
            }
            
            Log.d(tag, "特征统计信息加载成功")
        } catch (e: Exception) {
            Log.w(tag, "加载特征统计信息失败，将使用默认标准化: ${e.message}")
            // 如果加载失败，使用默认值
            featureMeans = FloatArray(inputDimension) { 0f }
            featureStds = FloatArray(inputDimension) { 1f }
        }
    }
    
    // 加载特征名称列表
    private fun loadFeatureNames() {
        try {
            val namesStream = context.assets.open("feature_names.json")
            val namesJson = namesStream.bufferedReader().use { it.readText() }
            val namesObject = JSONObject(namesJson)
            val namesArray = namesObject.getJSONArray("features")
            
            val names = mutableListOf<String>()
            for (i in 0 until namesArray.length()) {
                names.add(namesArray.getString(i))
            }
            
            featureNames = names
            Log.d(tag, "特征名称加载成功，共${names.size}个特征")
        } catch (e: Exception) {
            Log.w(tag, "加载特征名称失败: ${e.message}")
            featureNames = null
        }
    }

    private fun loadModels() {
        try {

            val modelName = "student_mobile.onnx"
            val modelAsset = context.assets.open(modelName)
            val modelFile = File(context.filesDir, modelName)
            
            modelAsset.use { input ->
                FileOutputStream(modelFile).use { output ->
                    input.copyTo(output)
                }
            }
            

            try {
                val correctorName = "sp_corrector.onnx"
                val correctorAsset = context.assets.open(correctorName)
                val correctorFile = File(context.filesDir, correctorName)
                
                correctorAsset.use { input ->
                    FileOutputStream(correctorFile).use { output ->
                        input.copyTo(output)
                    }
                }
                Log.d(tag, "收缩压修正器模型已复制")
            } catch (e: Exception) {
                Log.w(tag, "找不到收缩压修正器模型，跳过: ${e.message}")
            }
            
            Log.d(tag, "模型文件已复制到内部存储")
        } catch (e: Exception) {
            Log.e(tag, "复制模型文件失败", e)
        }
    }
    

    fun getInputDimension(): Int {
        return inputDimension
    }
    
    // 获取特征名称列表
    fun getFeatureNames(): List<String>? {
        return featureNames
    }
    
    // 标准化特征
    private fun standardizeFeatures(features: FloatArray): FloatArray {
        if (featureMeans == null || featureStds == null) {
            Log.w(tag, "标准化参数未加载，跳过标准化")
            return features
        }
        
        val standardized = FloatArray(features.size)
        for (i in features.indices) {
            if (i < featureMeans!!.size && i < featureStds!!.size) {
                val std = if (featureStds!![i] > 0.0001f) featureStds!![i] else 1f
                standardized[i] = (features[i] - featureMeans!![i]) / std
            } else {
                standardized[i] = features[i]
            }
        }
        
        return standardized
    }
    
    // 验证特征是否有异常值
    private fun validateFeatures(features: FloatArray): Boolean {
        var isValid = true
        var extremeValueCount = 0
        
        for (i in features.indices) {
            // 检查是否有极端值
            if (features[i].isNaN() || features[i].isInfinite()) {
                features[i] = 0f  // 替换无效值
                isValid = false
                extremeValueCount++
            }
            
            // 检查是否有过大或过小的值
            if (features[i] > 100f || features[i] < -100f) {
                extremeValueCount++
                // 不替换，只记录
            }
        }
        
        if (extremeValueCount > 0) {
            Log.w(tag, "特征中存在${extremeValueCount}个异常值")
        }
        
        return isValid
    }

    fun predict(features: FloatArray): Pair<Float, Float> {
        if (ortSession == null) {
            Log.e(tag, "模型未加载，无法预测")
            return Pair(120f, 80f) // 返回默认值
        }
        
        try {
            // 检查输入维度
            if (features.size != inputDimension) {
                Log.w(tag, "输入特征维度不匹配: ${features.size} vs $inputDimension")
                // 调整维度
                val adjustedFeatures = FloatArray(inputDimension)
                for (i in 0 until minOf(features.size, inputDimension)) {
                    adjustedFeatures[i] = features[i]
                }
                
                // 使用调整后的特征
                return predictInternal(adjustedFeatures)
            }
            
            return predictInternal(features)
        } catch (e: Exception) {
            Log.e(tag, "预测过程中发生错误", e)
            return Pair(120f, 80f) // 错误时返回默认值
        }
    }
    

    private fun predictInternal(features: FloatArray): Pair<Float, Float> {
        var inputTensor: OnnxTensor? = null
        var output: OrtSession.Result? = null
        var correctorInputTensor: OnnxTensor? = null
        var correctorOutput: OrtSession.Result? = null
        
        try {
            // 验证特征
            validateFeatures(features)
            
            // 标准化特征
            val standardizedFeatures = standardizeFeatures(features)
            
            // 记录特征统计信息
            logFeatureStats(standardizedFeatures)

            val inputName = ortSession!!.inputNames.iterator().next()
            val shape = longArrayOf(1, inputDimension.toLong())
            
            val inputBuffer = FloatBuffer.wrap(standardizedFeatures)
            inputTensor = OnnxTensor.createTensor(ortEnvironment, inputBuffer, shape)
            
            val inputs = mapOf(inputName to inputTensor)
            output = ortSession!!.run(inputs)
            
            @Suppress("UNCHECKED_CAST")
            val predictions = (output.get(0).value as Array<FloatArray>)[0]
            
            var systolic = predictions[0]
            var diastolic = predictions[1]
            
            if (spCorrectorSession != null) {
                try {
                    val correctorInputName = spCorrectorSession!!.inputNames.iterator().next()
                    val correctorShape = longArrayOf(1, 1)
                    val correctorInput = floatArrayOf(systolic)
                    val correctorInputBuffer = FloatBuffer.wrap(correctorInput)
                    correctorInputTensor = OnnxTensor.createTensor(ortEnvironment, correctorInputBuffer, correctorShape)
                    
                    val correctorInputs = mapOf(correctorInputName to correctorInputTensor)
                    correctorOutput = spCorrectorSession!!.run(correctorInputs)
                    
                    @Suppress("UNCHECKED_CAST")
                    val correctedSp = (correctorOutput.get(0).value as Array<FloatArray>)[0][0]
                    
                    systolic = correctedSp
                    Log.d(tag, "应用了收缩压修正: $systolic")
                } catch (e: Exception) {
                    Log.e(tag, "收缩压修正过程中发生错误", e)
                } finally {
                    // 释放收缩压修正器资源
                    correctorInputTensor?.close()
                    correctorOutput?.close()
                }
            }
            
            return Pair(systolic, diastolic)
        } catch (e: Exception) {
            Log.e(tag, "模型推理失败", e)
            return Pair(120f, 80f)
        } finally {
            // 确保所有资源都被释放
            inputTensor?.close()
            output?.close()
        }
    }
    
    // 记录特征统计信息，帮助调试
    private fun logFeatureStats(features: FloatArray) {
        var min = Float.MAX_VALUE
        var max = Float.MIN_VALUE
        var sum = 0f
        var zeroCount = 0
        
        for (feature in features) {
            if (feature < min) min = feature
            if (feature > max) max = feature
            sum += feature
            if (feature == 0f) zeroCount++
        }
        
        val mean = sum / features.size
        
        Log.d(tag, "特征统计: 最小值=$min, 最大值=$max, 平均值=$mean, 零值数量=$zeroCount/${features.size}")
    }
    
    // 辅助函数
    private fun min(first: Int, second: Int): Int {
        return if (first < second) first else second
    }

    fun close() {
        try {
            // 使用安全的关闭方式，确保即使某个资源关闭失败，其他资源仍然能被关闭
            try {
                ortSession?.close()
                ortSession = null
            } catch (e: Exception) {
                Log.e(tag, "关闭ortSession时发生错误", e)
            }
            
            try {
                spCorrectorSession?.close()
                spCorrectorSession = null
            } catch (e: Exception) {
                Log.e(tag, "关闭spCorrectorSession时发生错误", e)
            }
            
            try {
                ortEnvironment.close()
            } catch (e: Exception) {
                Log.e(tag, "关闭ortEnvironment时发生错误", e)
            }
            
            Log.d(tag, "所有ONNX资源已关闭")
        } catch (e: Exception) {
            Log.e(tag, "关闭资源时发生错误", e)
        }
    }
} 