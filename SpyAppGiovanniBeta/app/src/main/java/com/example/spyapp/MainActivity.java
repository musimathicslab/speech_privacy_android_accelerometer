package com.example.spyapp;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.media.MediaPlayer;
import android.media.MediaRecorder;
import android.os.Build;
import android.os.Bundle;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.IOException;
import java.util.Date;

public class MainActivity extends AppCompatActivity implements SensorEventListener {

    //Accellerometro
    private SensorManager sensorManager;
    Sensor accelerometer;

    //Gestione audio
    MediaRecorder recorder = null;
    private MediaPlayer player = null;
    private MediaPlayer player2 = null;
    String fileName = null;
    int counterFileName = 0;

    //Salvataggio audio
    Date createdTime = new Date();

    TextView nomeTraccia = null;
    TextView dataTraccia = null;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_NORMAL);

        fileName =  getExternalCacheDir().getAbsolutePath() + File.separator + "Nuova registrazione #" + counterFileName + File.separator + createdTime + File.separator + "audiorecordtest.3gp";

        System.out.println(fileName);

        nomeTraccia = (TextView) findViewById(R.id.traccia);
        dataTraccia = (TextView) findViewById(R.id.tracciaSotto);


    }

    @Override
    public void onSensorChanged(SensorEvent event) {
       /* if (flag) {
            t = event.timestamp;
            flag = false;
        }

        cont++;
        float[] linear_acceleration = new float[3];
        float tempo2 = (event.timestamp - t) / 1000000000f;


        linear_acceleration[0] = event.values[0];
        linear_acceleration[1] = event.values[1];
        linear_acceleration[2] = event.values[2];

        String scrivi = "" + tempo2 + "," + linear_acceleration[0] + "," + linear_acceleration[1] + "," + linear_acceleration[2] + "\n";
        try {
            stream.write(scrivi.getBytes());
        } catch (IOException e) {
            e.printStackTrace();
        }
    */
    }

    @Override //non ci serve
    public void onAccuracyChanged(Sensor sensor, int accuracy) { }

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    public void riproduci(View v) {
        player2 = new MediaPlayer();
        try {
            player2.setDataSource(fileName);
            player2.prepare();
            player2.start();
        } catch (IOException e) {
        }

    }

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    public void registra(View v) {
        Toast.makeText(this, "Start recording", Toast.LENGTH_LONG).show();
        recorder = new MediaRecorder();
        recorder.setAudioSource(MediaRecorder.AudioSource.MIC);
        recorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
        counterFileName++;
        recorder.setOutputFile(fileName);
        recorder.setAudioEncoder(MediaRecorder.AudioEncoder.AAC);
        recorder.setAudioEncodingBitRate(16*44100);
        recorder.setAudioSamplingRate(44100);
        try {
            recorder.prepare();
        } catch (IOException e) {
        }

        recorder.start();
    }

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    public void stopRegistrazione(View v) throws Exception {
        Toast.makeText(this, "Stopped reg", Toast.LENGTH_SHORT).show();
        recorder.stop();

        String fileNameAttuale = null;
        String dataAttuale = null;
        if(counterFileName < 9 && counterFileName > 0) {
            fileNameAttuale = fileName.substring(57, 80);
            dataAttuale = fileName.substring(82, 91) + " " + fileName.substring(112, 115);
        } else if(counterFileName > 9 && counterFileName < 999) {
            fileNameAttuale = fileName.substring(57, 81);
            dataAttuale = fileName.substring(83, 92) + " " + fileName.substring(113, 116);
        } else if(counterFileName < 0)
            throw new Exception("Negative number error!");
        else {
            fileNameAttuale = fileName.substring(57, 82);
            dataAttuale = fileName.substring(84, 93) + " " + fileName.substring(114, 117);
        }

        nomeTraccia.setText(fileNameAttuale);
        dataTraccia.setText(dataAttuale);
        recorder.release();
        recorder = null;
    }

}