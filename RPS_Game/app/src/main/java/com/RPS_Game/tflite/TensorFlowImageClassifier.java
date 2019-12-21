package com.RPS_Game.tflite;

import android.annotation.SuppressLint;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;


public class TensorFlowImageClassifier implements Classifier {

    private static final int MAX_RESULTS = 3; // Max tahmin sayısı
    private static final int BATCH_SIZE = 1; // Model Shape Gereği 1
    private static final int PIXEL_SIZE = 3; // Her piksel RGB = 3 channel var

    private Interpreter interpreter;
    private int inputSize;
    private List<String> labelList;



    static Classifier create(AssetManager assetManager,
                             String modelPath,
                             String labelPath,
                             int inputSize) throws IOException {

        TensorFlowImageClassifier classifier = new TensorFlowImageClassifier();
        //Modeli MappedByteBuffer olarak yüklüyoruz
        classifier.interpreter = new Interpreter(classifier.loadModelFile(assetManager, modelPath), new Interpreter.Options());
        //Sınıf isimleri List olarak yükleniyor
        classifier.labelList = classifier.loadLabelList(assetManager, labelPath);
        classifier.inputSize = inputSize;

        return classifier;
    }

    @Override
    public List<Recognition> recognizeImage(Bitmap bitmap) {
        // Modelimizi ByteBuffer olarak açtık, gelen resmi de kıyas için ByteBuffer'a çeviriyoruz
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmap);
        float [][] result = new float[1][labelList.size()];
        interpreter.run(byteBuffer, result);
        return getSortedResultFloat(result);


    }

    @Override
    public void close() {
        interpreter.close();
        interpreter = null;
    }

    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        //Asset Manager ile asset klasörü içindeki modelimizi açıyoruz
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        //FileDescriptor ile dosya tanımları okunuyor
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private List<String> loadLabelList(AssetManager assetManager, String labelPath) throws IOException {
        //Label text dosyamızın içindeki sınıf etiketleri List'e okunuyor
        List<String> labelList = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(assetManager.open(labelPath)));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer;

        byteBuffer = ByteBuffer.allocateDirect(4 * BATCH_SIZE * inputSize * inputSize * PIXEL_SIZE);


        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[inputSize * inputSize];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                final int val = intValues[pixel++];
                byteBuffer.putFloat((((val >> 16) & 0xFF)/255.0f));
                byteBuffer.putFloat((((val >> 8) & 0xFF)/255.0f));
                byteBuffer.putFloat((((val) & 0xFF)/255.0f));

            }
        }
        return byteBuffer;
    }


    @SuppressLint("DefaultLocale")
    private List<Recognition> getSortedResultFloat(float[][] labelProbArray) {

        PriorityQueue<Recognition> pq =
                new PriorityQueue<>(
                        MAX_RESULTS,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(Recognition lhs, Recognition rhs) {
                                return Float.compare(rhs.getRate(), lhs.getRate());
                            }
                        });

        for (int i = 0; i < labelList.size(); ++i) {
            float rate = labelProbArray[0][i];
            pq.add(new Recognition(labelList.get(i),rate));

        }

        final ArrayList<Recognition> recognitions = new ArrayList<>();
        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }

        return recognitions;
    }

}
