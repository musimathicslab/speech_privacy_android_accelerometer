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
    MediaRecorder recorder;
    private MediaPlayer player;
    String fileName;
    int counterFileName = 0;

    //Salvataggio audio
    Date createdTime;
    TextView nomeTraccia;
    TextView dataTraccia;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_NORMAL);

        //istanzia di nuovo per ottenere la data attuale e differenziarla nel TextView
        createdTime = new Date();
        fileName = getExternalCacheDir().getAbsolutePath() + File.separator + "Nuova Registrazione #" + counterFileName + " " + createdTime.toString().substring(0, 10) + " " + createdTime.toString().substring(30, 34) + createdTime.toString().substring(10, 19) + ".3gp";

        System.out.println(fileName);
        System.out.println(createdTime);

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
        player = new MediaPlayer();
        try {
            player.setDataSource(fileName);
            player.prepare();
            player.start();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    public void registra(View v) {
        Toast.makeText(this, "Start recording", Toast.LENGTH_SHORT).show();
        recorder = new MediaRecorder();
        recorder.setAudioSource(MediaRecorder.AudioSource.MIC);
        recorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
        createdTime = new Date();
        fileName = getExternalCacheDir().getAbsolutePath() + File.separator + "Nuova Registrazione #" + counterFileName + " " + createdTime.toString().substring(0, 10) + " " + createdTime.toString().substring(30, 34) + createdTime.toString().substring(10, 19) + ".3gp";
        counterFileName++;
        recorder.setOutputFile(fileName);
        recorder.setAudioEncoder(MediaRecorder.AudioEncoder.AAC);
        recorder.setAudioEncodingBitRate(16*44100);
        recorder.setAudioSamplingRate(44100);
        try {
            recorder.prepare();
        } catch (IOException e) {
            e.printStackTrace();
        }
        recorder.start();
    }

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    public void stopRegistrazione(View v) throws Exception {
        Toast.makeText(this, "Stopped reg", Toast.LENGTH_SHORT).show();
        recorder.stop();

        String fileNameAttuale, dataAttuale;

        fileNameAttuale = fileName.substring(58, 80);
        dataAttuale = fileName.substring(81, 105);

        nomeTraccia.setText(fileNameAttuale);
        dataTraccia.setText(dataAttuale);
        recorder.release();
        recorder = null;
    }

}