package com.rohitrj.catordog

import android.Manifest
import android.content.pm.PackageManager
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.util.Rational
import android.util.Size
import android.view.Surface
import android.view.TextureView
import android.view.ViewGroup
import android.widget.Toast
import androidx.appcompat.widget.Toolbar
import androidx.camera.core.*
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.android.material.appbar.CollapsingToolbarLayout
import com.google.firebase.ml.common.FirebaseMLException
import com.google.firebase.ml.common.modeldownload.FirebaseLocalModel
import com.google.firebase.ml.common.modeldownload.FirebaseModelDownloadConditions
import com.google.firebase.ml.common.modeldownload.FirebaseModelManager
import com.google.firebase.ml.common.modeldownload.FirebaseRemoteModel
import com.google.firebase.ml.custom.*
import com.google.firebase.ml.vision.FirebaseVision
import com.google.firebase.ml.vision.common.FirebaseVisionImage
import com.google.firebase.ml.vision.label.FirebaseVisionImageLabel
import com.google.firebase.ml.vision.label.FirebaseVisionImageLabeler
import com.google.firebase.ml.vision.label.FirebaseVisionOnDeviceAutoMLImageLabelerOptions
import kotlinx.android.synthetic.main.activity_main.*
import java.io.IOException
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.*
import kotlin.collections.ArrayList
import kotlin.experimental.and
import kotlin.math.abs


private const val REQUEST_CODE_PERMISSIONS = 10
private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)

val TAG = "MainActivity"
lateinit var mSelectedImage: Bitmap
lateinit var mGraphicOverlay: GraphicOverlay
// Max width (portrait mode)
var mImageMaxWidth: Int? = null
// Max height (portrait mode)
var mImageMaxHeight: Int? = null
/**
 * An instance of the driver class to run model inference with Firebase.
 */
var mInterpreter: FirebaseModelInterpreter? = null
var labeler: FirebaseVisionImageLabeler? = null
/**
 * Data configuration of input & output data of model.
 */
lateinit var mDataOptions: FirebaseModelInputOutputOptions
/**
 * Name of the model file hosted with Firebase.
 */
private val HOSTED_MODEL_NAME = "cat_or_dog_2019727221222"
private val LOCAL_MODEL_ASSET = "cat_or_dog.tflite"
/**
 * Name of the label file stored in Assets.
 */
private val LABEL_PATH = "labels.txt"
/**
 * Number of results to show in the UI.
 */
private val RESULTS_TO_SHOW = 1
/**
 * Dimensions of inputs.
 */
private val DIM_BATCH_SIZE = 1
private val DIM_PIXEL_SIZE = 3
private val DIM_IMG_SIZE_X = 200
private val DIM_IMG_SIZE_Y = 200
/**
 * Labels corresponding to the output of the vision model.
 */

private var mLabelList: List<String>? = null
private val intValues = IntArray(DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y)


class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        val toolbar = findViewById<Toolbar>(R.id.toolbar)
        setSupportActionBar(toolbar)
        val collapsingToolbar = findViewById<CollapsingToolbarLayout>(R.id.toolbar_layout)
        collapsingToolbar.title = "Cat Vs Dog"

        viewFinder = findViewById(R.id.camera_view)
        mGraphicOverlay = findViewById(R.id.graphicOverlay)


        // Request camera permissions
        if (allPermissionsGranted()) {
            viewFinder.post { startCamera() }
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }

        // Every time the provided texture view changes, recompute layout
        viewFinder.addOnLayoutChangeListener { _, _, _, _, _, _, _, _, _ ->
            updateTransform()
        }

        initCustomModel()

        camera_view.setOnClickListener {

            mSelectedImage = camera_view.bitmap
            runModelInference()
        }
    }

    private lateinit var viewFinder: TextureView

    private fun startCamera() {
        // Create configuration object for the viewfinder use case
        val previewConfig = PreviewConfig.Builder().apply {
            setTargetAspectRatio(Rational(1, 3))
            setTargetResolution(Size(400, 400))
        }.build()

        // Build the viewfinder use case
        val preview = Preview(previewConfig)

        // Every time the viewfinder is updated, recompute layout
        preview.setOnPreviewOutputUpdateListener {

            // To update the SurfaceTexture, we have to remove it and re-add it
            val parent = viewFinder.parent as ViewGroup
            parent.removeView(viewFinder)
            parent.addView(viewFinder, 0)

            viewFinder.surfaceTexture = it.surfaceTexture
            updateTransform()
        }

        // Bind use cases to lifecycle
        // If Android Studio complains about "this" being not a LifecycleOwner
        // try rebuilding the project or updating the appcompat dependency to
        // version 1.1.0 or higher.
        CameraX.bindToLifecycle(this, preview)
    }

    private fun updateTransform() {
        val matrix = Matrix()

        // Compute the center of the view finder
        val centerX = viewFinder.width / 2f
        val centerY = viewFinder.height / 2f

        // Correct preview output to account for display rotation
        val rotationDegrees = when (viewFinder.display.rotation) {
            Surface.ROTATION_0 -> 0
            Surface.ROTATION_90 -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else -> return
        }
        matrix.postRotate(-rotationDegrees.toFloat(), centerX, centerY)

        // Finally, apply transformations to our TextureView
        viewFinder.setTransform(matrix)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                viewFinder.post { startCamera() }
            } else {
                Toast.makeText(
                    this,
                    "Permissions not granted by the user.",
                    Toast.LENGTH_SHORT
                ).show()
                finish()
            }
        }
    }

    /**
     * Check if all permission specified in the manifest have been granted
     */
    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it
        ) == PackageManager.PERMISSION_GRANTED
    }


    private fun initCustomModel() {

        mLabelList = loadLabelList()
        val inputDims = intArrayOf(DIM_BATCH_SIZE, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y, DIM_PIXEL_SIZE)
        val outputDims = intArrayOf(DIM_BATCH_SIZE, mLabelList!!.size)

        try {
            mDataOptions = FirebaseModelInputOutputOptions.Builder()
                .setInputFormat(0, FirebaseModelDataType.BYTE, inputDims)
                .setOutputFormat(0, FirebaseModelDataType.BYTE, outputDims)
                .build()

            val conditions: FirebaseModelDownloadConditions = FirebaseModelDownloadConditions
                .Builder()
                .build()

            val remoteModel: FirebaseRemoteModel = FirebaseRemoteModel.Builder(HOSTED_MODEL_NAME)
                .enableModelUpdates(true)
                .setInitialDownloadConditions(conditions)
                .setUpdatesDownloadConditions(conditions)  // You could also specify
                // different conditions
                // for updates
                .build()

            val localModel: FirebaseLocalModel = FirebaseLocalModel.Builder("asset")
                .setAssetFilePath(LOCAL_MODEL_ASSET).build()

            val manager: FirebaseModelManager = FirebaseModelManager.getInstance()
            manager.registerRemoteModel(remoteModel)
            manager.registerLocalModel(localModel)

//            val modelOptions: FirebaseModelOptions = FirebaseModelOptions.Builder()
//                .setRemoteModelName(HOSTED_MODEL_NAME)
//                .setLocalModelName("asset")
//                .build()
            val modelOptions = FirebaseVisionOnDeviceAutoMLImageLabelerOptions.Builder()
                .setConfidenceThreshold(.8f)
                .setRemoteModelName(HOSTED_MODEL_NAME)
                .setLocalModelName("asset")
                .build()
                
            labeler = FirebaseVision.getInstance().getOnDeviceAutoMLImageLabeler(modelOptions)

//            mInterpreter = FirebaseModelInterpreter.getInstance(modelOptions)!!


        } catch (e: FirebaseMLException) {
            showToast("Error while setting up the model")
            e.printStackTrace()
        }

    }

    private fun runModelInference() {
        mGraphicOverlay.clear()
//        if (mInterpreter == null) {
//            Log.e(TAG, "Image classifier has not been initialized; Skipped.")
//            return
//        }
//        // Create input data.
//        val imgData: ByteBuffer = convertBitmapToByteBuffer(mSelectedImage)
//
//        try {
//            val inputs: FirebaseModelInputs = FirebaseModelInputs.Builder()
//                .add(imgData).build()
//            // Here's where the magic happens!!
//            mInterpreter!!.run(inputs, mDataOptions)
//                .addOnFailureListener {
//                    it.printStackTrace()
//                    showToast("Error running model inference")
//                }
//                .continueWith {
//                    //                    Log.i(TAG,it.result!!.getOutput(0))
//                    val labelProbArray: Array<ByteArray> = it.result!!.getOutput(0)
//                    val topLabels: List<String> = getTopLabels(labelProbArray)
//                    val labelGraphic: GraphicOverlay.Graphic =
//                        LabelGraphic(mGraphicOverlay, topLabels)
//                    mGraphicOverlay.add(labelGraphic)
//                    return@continueWith topLabels
//                }
//        } catch (e: FirebaseMLException) {
//            e.printStackTrace()
//            showToast("Error  exception running model inference")
//        }


        if (labeler == null) {
            Log.e(TAG, "Image classifier has not been initialized; Skipped.")
            return
        }
        val image = FirebaseVisionImage.fromBitmap(mSelectedImage)

        labeler!!.processImage(image).continueWith { task ->

            val labelProbList = task.result

            // Indicate whether the remote or local model is used.
            // Note: in most common cases, once a remote model is downloaded it will be used. However, in
            // very rare cases, the model itself might not be valid, and thus the local model is used. In
            // addition, since model download failures can be transient, and model download can also be
            // triggered in the background during inference, it is possible that a remote model is used
            // even if the first download fails.
            val textToShow =  if (labelProbList.isNullOrEmpty())
                "No Result"
            else
                printTopKLabels(labelProbList)

            var topLabels = listOf(textToShow)

            Log.i("labels", textToShow)
            val labelGraphic: GraphicOverlay.Graphic =
                        LabelGraphic(mGraphicOverlay, topLabels)
                    mGraphicOverlay.add(labelGraphic)

            // print the results
            textToShow
        }

    }

    /** Prints top-K labels, to be shown in UI as the results.  */
    private val printTopKLabels: (List<FirebaseVisionImageLabel>) -> String = {
        it.joinToString(
            separator = "\n",
            limit = RESULTS_TO_SHOW
        ) { label ->
            String.format(Locale.getDefault(), "Label: %s, Confidence: %4.2f", label.text, label.confidence)
        }
    }

    @Synchronized
    private fun getTopLabels(labelProbArray: Array<ByteArray>): List<String> {

        val result: ArrayList<String> = ArrayList()

        for (i in mLabelList!!.indices) {

            Log.i(
                "labels",
                "${mLabelList?.get(i)}: ${(labelProbArray[0][i] / 255.0f)}"
            )
//            val prob = (labelProbArray[0][i] and 0xff.toByte()) / 255.0f
        }


//        val catProb = (labelProbArray[0][0] and 0xff.toByte()) / 255.0f
//        val dogProb = (labelProbArray[0][1] and 0xff.toByte()) / 255.0f
//        Log.i("labels", "cat: $catProb")
//        Log.i("labels", "dog: $dogProb")

        return result
    }


    /**
     * Writes Image data into a `ByteBuffer`.
     */
    @Synchronized
    private fun convertBitmapToByteBuffer(
        bitmap: Bitmap
    ): ByteBuffer {
        val imgData = ByteBuffer.allocateDirect(
            DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE
        )
        imgData.order(ByteOrder.nativeOrder())
        val scaledBitmap = Bitmap.createScaledBitmap(
            bitmap, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y,
            true
        )
        imgData.rewind()
        scaledBitmap.getPixels(
            intValues, 0, scaledBitmap.width, 0, 0,
            scaledBitmap.width, scaledBitmap.height
        )
        // Convert the image to int points.
        var pixel = 0
        for (i in 0 until DIM_IMG_SIZE_X) {
            for (j in 0 until DIM_IMG_SIZE_Y) {
                val `val` = intValues[pixel++]
                imgData.put((`val` shr 16 and 0xFF).toByte())
                imgData.put((`val` shr 8 and 0xFF).toByte())
                imgData.put((`val` and 0xFF).toByte())
            }
        }
        return imgData
    }


    private fun showToast(message: String) {
        Toast.makeText(applicationContext, message, Toast.LENGTH_SHORT).show()
    }

    /**
     * Reads label list from Assets.
     */
    private fun loadLabelList(): List<String> {
        val labelList = ArrayList<String>()
        val assetManager: AssetManager = resources.assets
        val inputStream: InputStream?
        try {
            inputStream = assetManager.open(LABEL_PATH)
            val s = Scanner(inputStream)
            while (s.hasNext()) {
//                Log.i("labelstxt",s.nextLine() )
                labelList.add(s.nextLine())
            }

        } catch (e: IOException) {
            e.printStackTrace()
        }
        return labelList
    }


}
