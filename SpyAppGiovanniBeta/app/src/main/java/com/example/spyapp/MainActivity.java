package com.example.spyapp;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.media.Image;
import android.media.MediaPlayer;
import android.media.MediaRecorder;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.FileDescriptor;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Date;

public class MainActivity extends AppCompatActivity implements SensorEventListener {

    //Accellerometro
    private SensorManager sensorManager;
    Sensor accelerometer;
    boolean accellerometerFlag;
    ImageView eyeAccellerometer;
    /*long t;
    int cont = 0;*/

    //storage
    File directory;

    //Gestione audio
    MediaRecorder recorder;
    MediaPlayer player;
    String fileName;
    ArrayList<File> fileNames;
    int counterFileName;
    int adj = 1;
    FileDescriptor input;

    //Salvataggio audio
    Date createdTime;
    TextView nomeTraccia;
    TextView dataTraccia;

    //console
    TextView console;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        accellerometerFlag = true;

        eyeAccellerometer = (ImageView) findViewById(R.id.controlloAccellerometro);

        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_FASTEST);

        //istanzia di nuovo per ottenere la data attuale e differenziarla nel TextView
        createdTime = new Date();

        fileName = getExternalCacheDir().getAbsolutePath() + File.separator + "Nuova Registrazione #" + counterFileName + " " + createdTime.toString().substring(0, 10) + " " + createdTime.toString().substring(30, 34) + createdTime.toString().substring(10, 19) + ".3gp";
        fileNames = new ArrayList<File>();
        directory = getExternalCacheDir();

        for(File file : directory.listFiles())
            fileNames.add(file);

        counterFileName = directory.listFiles().length;

        nomeTraccia = (TextView) findViewById(R.id.traccia);
        dataTraccia = (TextView) findViewById(R.id.tracciaSotto);

        console = (TextView) findViewById(R.id.textBoxConsole);

        if(counterFileName > 0){
            setRegistrazione(fileNames.get(counterFileName - adj).toString());
        } else {
            nomeTraccia.setText("Nessuna traccia audio");
            dataTraccia.setText(" ");

        }
    }

    //Accelerometro si ferma quando l'app è in background per evitare consumi inutili di batteria
    @Override
    protected void onPause() {
        super.onPause();
        sensorManager.unregisterListener(this);
    }

    @Override
    protected void onResume() {
        super.onResume();
        sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_NORMAL);
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
      /* if (accellerometerFlag) {
            t = event.timestamp;
            accellerometerFlag = false;
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
        }*/
    }

    @Override //non ci serve
    public void onAccuracyChanged(Sensor sensor, int accuracy) { }

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    public void riproduci(View v) {
        player = new MediaPlayer();
        try{
            player.setDataSource(this, Uri.parse(fileName));
            player.prepare();
            player.start();
        } catch(Exception e){
            e.printStackTrace();
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    public void registra(View v) {
        Toast.makeText(this, "Start recording", Toast.LENGTH_SHORT).show();
        recorder = new MediaRecorder();
        recorder.setAudioSource(MediaRecorder.AudioSource.MIC);
        recorder.setOutputFormat(MediaRecorder.OutputFormat.THREE_GPP);
        createdTime = new Date();
        fileName = getExternalCacheDir().getAbsolutePath() + File.separator + "Nuova Registrazione #" + counterFileName + " " + createdTime.toString().substring(0, 10) + " " + createdTime.toString().substring(30, 34) + createdTime.toString().substring(10, 19) + ".3gp";
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

       /* if(accellerometerFlag){
            try{
                stream = openFileOutput("dataRegistrazione" + counterFileName + ".csv", Context.MODE_PRIVATE);
                String scrivi = "TIME," + "X," + "Y," + "Z\n";
                stream.write(scrivi.getBytes());
                store();

            } catch (Exception e) {
                e.printStackTrace();
            }
        } else if(!accellerometerFlag){

        }*/

    }

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    public void stopRegistrazione(View v) throws Exception {
        try {
            recorder.stop();
            Toast.makeText(this, "Stopped reg", Toast.LENGTH_SHORT).show();

            setRegistrazione(fileName);

            recorder.release();
            recorder = null;

            fileNames = new ArrayList<File>();
            directory = getExternalCacheDir();

            for (File file : directory.listFiles())
                fileNames.add(file);

            counterFileName = directory.listFiles().length;
        } catch(Exception e){
            e.printStackTrace();
            Toast.makeText(this, "No recording in progress!", Toast.LENGTH_SHORT).show();
        }
    }

    public void controlAccelerometro(View view) {
        if(accellerometerFlag){
            Toast.makeText(this, "Accellerometer: OFF", Toast.LENGTH_SHORT).show();
            console.setText("Accellerometer: OFF");
            eyeAccellerometer.setImageResource(R.drawable.eyeclosed);
            sensorManager.unregisterListener(this);
            accellerometerFlag = false;
        } else if(!accellerometerFlag){
            Toast.makeText(this, "Accellerometer: ON", Toast.LENGTH_SHORT).show();
            console.setText("Accellerometer: ON");
            eyeAccellerometer.setImageResource(R.drawable.eyeopen);
            sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_FASTEST);
            accellerometerFlag = true;
        }
    }

    public void setRegistrazione(String fileName){
        String fileNameAttuale, dataAttuale;

        fileNameAttuale = fileName.substring(58, 80);
        dataAttuale = fileName.substring(81, 105);

        nomeTraccia.setText(fileNameAttuale);
        dataTraccia.setText(dataAttuale);
    }

    public void skipRight(View view) {
        try {
            adj--;
            setRegistrazione(fileNames.get(counterFileName - adj).toString());
            fileName = fileNames.get(counterFileName - adj).getAbsolutePath();
            System.out.println(fileName);
        } catch(Exception e){
            e.printStackTrace();
            Toast.makeText(this, "No more audio files", Toast.LENGTH_SHORT).show();
            adj++;
        }
    }

    public void skipLeft(View view) {
        try {
            adj++;
            setRegistrazione(fileNames.get(counterFileName - adj).toString());
            fileName = fileNames.get(counterFileName - adj).getAbsolutePath();
            System.out.println(fileName);
        } catch(Exception e){
            e.printStackTrace();
            Toast.makeText(this, "No more audio files", Toast.LENGTH_SHORT).show();
            adj--;
        }
    }
}