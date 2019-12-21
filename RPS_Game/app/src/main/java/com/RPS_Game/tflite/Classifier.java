package com.RPS_Game.tflite;

import android.graphics.Bitmap;

import java.util.List;


public interface Classifier {

    class Recognition {
        private final String title;
        private final Float rate;


        public Recognition(final String title, final Float rate) {
            this.title = title;
            this.rate = rate;

        }
        public String getTitle() {
            return title;
        }

        public Float getRate() {
            return rate;
        }

        @Override
        public String toString() {
            String resultString = "";
            if (title != null) {
                resultString += title + " ";
            }

            if (rate != null) {
                resultString += String.format("(%.1f%%) ", rate * 100.0f);
            }

            return resultString.trim();
        }
    }


    List<Recognition> recognizeImage(Bitmap bitmap);

    void close();
}
