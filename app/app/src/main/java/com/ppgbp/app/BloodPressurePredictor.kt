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
    private var featureMeans: FloatArray? = null
    private var featureStds: FloatArray? = null
    private var featureNames: List<String>? = null
    
    init {
        try {

            loadModels()

            // Load feature statistics
            loadFeatureStats()


            // Load feature names
            loadFeatureNames()
            

            val sessionOptions = OrtSession.SessionOptions()
            sessionOptions.setIntraOpNumThreads(2)
            

            val modelFile = File(context.filesDir, "student_mobile.onnx")
            ortSession = ortEnvironment.createSession(modelFile.absolutePath, sessionOptions)
            

            val spCorrectorFile = File(context.filesDir, "sp_corrector.onnx")
            if (spCorrectorFile.exists()) {
                Log.d(tag, "Load systolic blood pressure corrector model")
                spCorrectorSession = ortEnvironment.createSession(spCorrectorFile.absolutePath, sessionOptions)
            }
            
            Log.d(tag, " Model loaded successfully")
        } catch (e: Exception) {
            Log.e(tag, "Model loading failed", e)
        }
    }

    // Load feature statistics (mean and standard deviation)
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
            
            Log.d(tag, "Feature statistics loaded successfully")
        } catch (e: Exception) {
            Log.w(tag, "Failed to load feature statistics - default normalization will be applied: ${e.message}")
            //Fallback to default values if loading fails
            featureMeans = FloatArray(inputDimension) { 0f }
            featureStds = FloatArray(inputDimension) { 1f }
        }
    }


    // Load feature name list
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
            Log.d(tag, "Feature names loaded successfully, total ${names.size} features")
        } catch (e: Exception) {
            Log.w(tag, "Failed to load feature names ${e.message}")
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
                Log.d(tag, "Systolic blood pressure corrector model copied")
            } catch (e: Exception) {
                Log.w(tag, "Systolic blood pressure corrector model not found - skipping correction: ${e.message}")
            }
            
            Log.d(tag, "the model file has been copied to internal storage")
        } catch (e: Exception) {
            Log.e(tag, "failed to copy the model file", e)
        }
    }
    

    fun getInputDimension(): Int {
        return inputDimension
    }

    fun getFeatureNames(): List<String>? {
        return featureNames
    }
    
    // Normalize features
    private fun standardizeFeatures(features: FloatArray): FloatArray {
        if (featureMeans == null || featureStds == null) {
            Log.w(tag, "Normalization parameters not loaded - skipping normalization")
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
    
    // Check if the features have outliers
    private fun validateFeatures(features: FloatArray): Boolean {
        var isValid = true
        var extremeValueCount = 0
        
        for (i in features.indices) {

            //Check for extreme values
            if (features[i].isNaN() || features[i].isInfinite()) {
                features[i] = 0f
                isValid = false
                extremeValueCount++
            }
            
            // Check for values that are too large or too small
            if (features[i] > 100f || features[i] < -100f) {
                extremeValueCount++
                // Do not replace, only record.
            }
        }
        
        if (extremeValueCount > 0) {
            Log.w(tag, "There are ${extremeValueCount} outlier(s) in the features.")
        }
        
        return isValid
    }

    fun predict(features: FloatArray): Pair<Float, Float> {
        if (ortSession == null) {
            Log.e(tag, "Model not loaded, prediction cannot be performed.")
            return Pair(120f, 80f) // Return default value.
        }
        
        try {

            if (features.size != inputDimension) {
                Log.w(tag, "Input feature dimensions do not match: ${features.size} vs $inputDimension")
                val adjustedFeatures = FloatArray(inputDimension)
                for (i in 0 until minOf(features.size, inputDimension)) {
                    adjustedFeatures[i] = features[i]
                }
                

                return predictInternal(adjustedFeatures)
            }
            
            return predictInternal(features)
        } catch (e: Exception) {
            Log.e(tag, "An error occurred during the prediction process.", e)
            return Pair(120f, 80f)
        }
    }
    

    private fun predictInternal(features: FloatArray): Pair<Float, Float> {
        var inputTensor: OnnxTensor? = null
        var output: OrtSession.Result? = null
        var correctorInputTensor: OnnxTensor? = null
        var correctorOutput: OrtSession.Result? = null
        
        try {

            validateFeatures(features)
            

            val standardizedFeatures = standardizeFeatures(features)
            

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
                    Log.d(tag, "Systolic blood pressure correction has been applied: $systolic")
                } catch (e: Exception) {
                    Log.e(tag, "An error occurred during systolic blood pressure correction", e)
                } finally {
                    // Release systolic blood pressure corrector resources
                    correctorInputTensor?.close()
                    correctorOutput?.close()
                }
            }
            
            return Pair(systolic, diastolic)
        } catch (e: Exception) {
            Log.e(tag, "Model inference failed", e)
            return Pair(120f, 80f)
        } finally {

            inputTensor?.close()
            output?.close()
        }
    }
    

    //Log feature statistics to assist with debugging
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
        
        Log.d(tag, "Feature statistics: min=$min, max=$max, mean=$mean, zero count=$zeroCount/${features.size}")
    }
    

    //Helper function
    private fun min(first: Int, second: Int): Int {
        return if (first < second) first else second
    }

    fun close() {
        try {
            // Use a safe shutdown approach to ensure that even if one resource fails to close, other resources can still be closed properly.
            try {
                ortSession?.close()
                ortSession = null
            } catch (e: Exception) {
                Log.e(tag, "An error occurred while closing the ortSession.", e)
            }
            
            try {
                spCorrectorSession?.close()
                spCorrectorSession = null
            } catch (e: Exception) {
                Log.e(tag, "An error occurred while closing the spCorrectorSession.", e)
            }
            
            try {
                ortEnvironment.close()
            } catch (e: Exception) {
                Log.e(tag, "An error occurred while closing the ortEnvironment.", e)
            }
            
            Log.d(tag, "All ONNX resources have been closed.")
        } catch (e: Exception) {
            Log.e(tag, "An error occurred while closing the resources.", e)
        }
    }
} 
