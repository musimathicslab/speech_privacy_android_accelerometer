package com.example.spyapp;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.media.AudioManager;
import android.media.Image;
import android.media.MediaPlayer;
import android.media.MediaRecorder;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.view.Window;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ViewSwitcher;

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
    boolean accellerometerFlagSeconds = true;
    ImageView eyeAccellerometer;
    long t;
    int cont = 0;
    int personalCounter = 0;
    FileOutputStream stream;

    //storage
    File directory;
    SharedPreferences preferences;
    SharedPreferences.Editor editor;

    //Gestione audio
    MediaRecorder recorder;
    MediaPlayer player;
    String fileName;
    ArrayList<File> fileNames;
    int counterFileName;
    int adj = 1;
    FileDescriptor input;
    AudioManager audioChecker;

    //Salvataggio audio
    Date createdTime;
    TextView nomeTraccia;
    TextView dataTraccia;

    //console
    TextView console;

    //View centrale
    ImageView centrale;
    boolean wave;

    //Audio pre-caricato
    boolean precaricato = false;
    int[] speechUnit = {R.raw.a1,R.raw.a2,R.raw.a3,R.raw.a4,R.raw.a5,R.raw.all1,R.raw.all2,R.raw.all3,
            R.raw.all4,R.raw.all5,R.raw.as1,R.raw.as2,R.raw.as3,R.raw.as4,R.raw.as5,R.raw.be1,
            R.raw.be2,R.raw.be3,R.raw.be4,R.raw.be5,R.raw.beau1,R.raw.beau2,R.raw.beau3,R.raw.beau4,
            R.raw.beau5,R.raw.best1,
            R.raw.best2,R.raw.best3,R.raw.best4,R.raw.best5,R.raw.birds1,R.raw.birds2,R.raw.birds3,
            R.raw.birds4,R.raw.birds5,R.raw.bite1,R.raw.bite2,R.raw.bite3,R.raw.bite4,R.raw.bite5,
            R.raw.book1,R.raw.book2,R.raw.book3,R.raw.book4,R.raw.book5,R.raw.but1,R.raw.but2,
            R.raw.but3,R.raw.but4,R.raw.but5,R.raw.by1,R.raw.by2,
            R.raw.by3,R.raw.by4,R.raw.by5,R.raw.can1,R.raw.can2,R.raw.can3,R.raw.can4,R.raw.can5,
            R.raw.cant1,R.raw.cant2,R.raw.cant3,R.raw.cant4,R.raw.cant5,R.raw.co1,R.raw.co2,R.raw.co3,
            R.raw.co4,R.raw.co5,R.raw.cy1,R.raw.cy2,R.raw.cy3,R.raw.cy4,R.raw.cy5,R.raw.day1,
            R.raw.day2,R.raw.day3,R.raw.day4,R.raw.day6,R.raw.der1,R.raw.der2,R.raw.der3,R.raw.der4,
            R.raw.der5,R.raw.do1,R.raw.do2,R.raw.do3,R.raw.do4, R.raw.do5,R.raw.done1,R.raw.done2,
            R.raw.done3,R.raw.done4,R.raw.done5,R.raw.dont1,R.raw.dont2,R.raw.dont3,R.raw.dont4,
            R.raw.dont5,R.raw.drink,R.raw.drink2,R.raw.drink3,R.raw.drink4,R.raw.drink5,R.raw.eye1,
            R.raw.eye2,R.raw.eye3,R.raw.eye4,R.raw.eye5,R.raw.fea1,R.raw.fea2,R.raw.fea3,R.raw.fea4,
            R.raw.fea5,R.raw.feeds1,R.raw.feeds2,R.raw.feeds3,R.raw.feeds4,R.raw.feeds5,R.raw.fire1,
            R.raw.fire2,R.raw.fire3,R.raw.fire4,R.raw.fire5,R.raw.flock1,R.raw.flock2,R.raw.flock3,
            R.raw.flock4,R.raw.flock5,R.raw.free1,R.raw.free2,R.raw.free3,R.raw.free4,R.raw.free5,
            R.raw.get1,R.raw.get2,R.raw.get3,R.raw.get4,R.raw.get5,R.raw.gether1,R.raw.gether2,
            R.raw.gether3,R.raw.gether4,R.raw.gether5,R.raw.hand1,R.raw.hand2,R.raw.hand3,R.raw.hand4,
            R.raw.hand5,R.raw.have1,R.raw.have2,R.raw.have3,R.raw.have4,R.raw.have5,R.raw.him1,
            R.raw.him2,R.raw.him3,R.raw.him4,R.raw.him5,R.raw.hol1,R.raw.hol2,R.raw.hol3,R.raw.hol4,
            R.raw.hol5,R.raw.home1,R.raw.home2,R.raw.home3,R.raw.home4,R.raw.home5,R.raw.horse1,
            R.raw.horse2,R.raw.horse3,R.raw.horse5,R.raw.horse7,R.raw.i1,R.raw.i2,R.raw.i3,R.raw.i4,
            R.raw.i5,R.raw.if1,R.raw.if2,R.raw.if3,R.raw.if4,R.raw.if5,R.raw.in1,R.raw.in2,R.raw.in3,
            R.raw.in4,R.raw.in5,R.raw.is1,R.raw.is2,R.raw.is3,R.raw.is4,R.raw.is5,R.raw.it1,R.raw.it2,
            R.raw.it3,R.raw.it4,R.raw.it5,R.raw.its1,R.raw.its2,R.raw.its3,R.raw.its4,R.raw.its5,
            R.raw.judge1,R.raw.judge2,R.raw.judge3,R.raw.judge4,R.raw.judge5,R.raw.lead1,R.raw.lead2,
            R.raw.lead3,R.raw.lead4,R.raw.lead5,R.raw.li1,R.raw.li2,R.raw.li3,R.raw.li4,R.raw.li5,
            R.raw.like1,R.raw.like2,R.raw.like3, R.raw.like4,R.raw.like5,R.raw.lunch1,R.raw.lunch4,
            R.raw.lunch5,R.raw.lunch6,R.raw.lunch7,R.raw.ma1,R.raw.ma2,R.raw.ma3,R.raw.ma4,R.raw.ma5,
            R.raw.make1,R.raw.make2,R.raw.make3,R.raw.make4,R.raw.make5,R.raw.mo1,R.raw.mo2,R.raw.mo3,
            R.raw.mo4,R.raw.mo5,R.raw.ne1,R.raw.ne2,R.raw.ne3,R.raw.ne4,R.raw.ne5,R.raw.no1,R.raw.no2,
            R.raw.no3,R.raw.no4,R.raw.no5,R.raw.ny1,R.raw.ny2,R.raw.ny3,R.raw.ny4,R.raw.ny5,R.raw.o1,
            R.raw.o2,R.raw.o3,R.raw.o4,R.raw.o5,R.raw.of1,R.raw.of2,R.raw.of3,R.raw.of4,R.raw.of5,
            R.raw.off1,R.raw.off2,R.raw.off3,R.raw.off4,R.raw.off5,R.raw.place1,R.raw.place2,
            R.raw.place3,R.raw.place4,R.raw.place5,R.raw.po1,R.raw.po2,R.raw.po3,R.raw.po4,R.raw.po5,
            R.raw.pre1,R.raw.pre2,R.raw.pre3,R.raw.pre4,R.raw.pre5,R.raw.put1,R.raw.put2,R.raw.put3,
            R.raw.put4,R.raw.put5,R.raw.right1,R.raw.right2,R.raw.right3,R.raw.right4,R.raw.right5,
            R.raw.rons,R.raw.rons2,R.raw.rons3,R.raw.rons4,R.raw.rons5,R.raw.rrow1,R.raw.rrow2,
            R.raw.rrow3,R.raw.rrow4,R.raw.rrow5,R.raw.self1,R.raw.self2,R.raw.self3,R.raw.self4,
            R.raw.self5,R.raw.sent1,R.raw.sent2,R.raw.sent3,R.raw.sent4,R.raw.sent5,R.raw.some1,
            R.raw.some2,R.raw.some3,R.raw.some4,R.raw.some5,R.raw.sty1,R.raw.sty2,R.raw.sty3,R.raw.sty4,
            R.raw.sty5,R.raw.such1,R.raw.such2,R.raw.such3,R.raw.such4,R.raw.such5,R.raw.ter1,
            R.raw.ter2,R.raw.ter3,R.raw.ter4,R.raw.ter5,R.raw.that1,R.raw.that2,R.raw.that3,
            R.raw.that4,R.raw.that5,R.raw.the1,R.raw.the2,R.raw.the3,R.raw.the4,R.raw.the5,R.raw.ther1,
            R.raw.ther2,R.raw.ther3,R.raw.ther4,R.raw.ther5,R.raw.there1,R.raw.there2,R.raw.there3,R.raw.there4,
            R.raw.there5,R.raw.thing1,R.raw.thing2,R.raw.thing,R.raw.thing4,R.raw.thing5,R.raw.til1,
            R.raw.til2,R.raw.til3,R.raw.til4,R.raw.til5,R.raw.time1,R.raw.time2,R.raw.time3,R.raw.time4,
            R.raw.time5,R.raw.to1,R.raw.to2,R.raw.to3,R.raw.to4,R.raw.to5,R.raw.too1,R.raw.too2,
            R.raw.too3,R.raw.too4,R.raw.too5,R.raw.ty1,R.raw.ty2,R.raw.ty3,R.raw.ty4,R.raw.ty5,
            R.raw.un1,R.raw.un2,R.raw.un3,R.raw.un4,R.raw.un5,R.raw.ver1,R.raw.ver2,R.raw.ver3,
            R.raw.ver4,R.raw.ver5,R.raw.wa1,R.raw.wa2,R.raw.wa3,R.raw.wa4,R.raw.wa5,R.raw.want1,
            R.raw.want2,R.raw.want3,R.raw.want4,R.raw.want5,R.raw.ways1,R.raw.ways2,R.raw.ways3,
            R.raw.ways4,R.raw.ways5,R.raw.what1,R.raw.what2,R.raw.what3,R.raw.what4,R.raw.what5,
            R.raw.you1,R.raw.you2,R.raw.you3,R.raw.you4,R.raw.you5,R.raw.your1,R.raw.your2,
            R.raw.your3,R.raw.your4,R.raw.your5,R.raw.frase1,R.raw.frase2,R.raw.frase3,R.raw.frase4,
            R.raw.frase5,R.raw.frase12,R.raw.frase121,R.raw.frase122,R.raw.frase123,R.raw.frase13,
            R.raw.frase131,R.raw.frase132,R.raw.frase133,R.raw.frase14,R.raw.frase141,R.raw.frase142,
            R.raw.frase143,R.raw.frase18,R.raw.frase19,R.raw.frase20};

    String[] speech = {"a","all","as","be","beau","best","birds","bite","book","but","by","can","cant","co","cy", "day","der", "do", "done","dont","drink", "eye","fea","feeds","fire","flock","free","get","gether","hand", "have","him","hol", "home", "horse",
            "i","if","in", "is", "it","its","judge","lead","li", "like", "lunch","ma","make","mo","ne","no","ny","o","of","off", "place","po","pre","put", "right","rons","rrow", "self","sent", "some","sty","such","ter","that","the","ther",
            "there", "thing","til","time", "to","too","ty","un","ver","wa", "want", "ways", "what", "you", "your","frasi1","frasi2","frasi3","frasi4"};
    String[] vocal = {"1","2","3","4","5"};
    int posizione, risorsa;

    //Set ip and port
    EditText ipEdit;
    EditText portEdit;
    Button setButton;
    String ip, port;

    @RequiresApi(api = Build.VERSION_CODES.M)
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        requestPermissions(new String[]{ Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.RECORD_AUDIO,Manifest.permission.INTERNET}, PackageManager.PERMISSION_GRANTED);

        refresh();

        audioChecker = (AudioManager) getSystemService(Context.AUDIO_SERVICE);

        preferences = getSharedPreferences("generalSettings", Context.MODE_PRIVATE);
        editor = preferences.edit();

        posizione = preferences.getInt("pos", 0);

        ip = preferences.getString("IP", null);
        port = preferences.getString("PORT", null);

        accellerometerFlag = true;

        eyeAccellerometer = (ImageView) findViewById(R.id.controlloAccellerometro);

        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_FASTEST);

        //istanzia di nuovo per ottenere la data attuale e differenziarla nel TextView
        createdTime = new Date();

        nomeTraccia = (TextView) findViewById(R.id.traccia);
        dataTraccia = (TextView) findViewById(R.id.tracciaSotto);

        centrale = (ImageView) findViewById(R.id.soundWaves);

        console = (TextView) findViewById(R.id.textBoxConsole);

        if(!precaricato) {
            if (counterFileName > 0) {
                fileName = fileNames.get(counterFileName - adj).toString();
                setRegistrazione(fileName, counterFileName);
            } else {
                nomeTraccia.setText("Nessuna traccia audio");
                dataTraccia.setText(" ");
                centrale.setImageResource(R.drawable.privacy);
                centrale.setMaxHeight(99);
                centrale.setMaxWidth(231);
                wave = false;
            }
        } else {
            setRegistrazione(null, 0);
        }

        try {
            stream = openFileOutput("data" + (counterFileName - adj) + "_" + personalCounter + ".csv", Context.MODE_PRIVATE);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
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

    @Override //non ci serve
    public void onAccuracyChanged(Sensor sensor, int accuracy) { }

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    public void riproduci(View v) {
        if(!precaricato) {
            player = new MediaPlayer();
            try {
                player.setDataSource(fileName);
                player.prepare();
                player.start();
            } catch (Exception e) {
                e.printStackTrace();
            }
        } else {
            player = MediaPlayer.create(getApplicationContext(), risorsa);
            if(!player.isPlaying()) {
                player.start();
            }
        }
        new TaskBackground().execute(0);
        personalCounter++;
        try {
            stream.close();
        } catch (Exception e){
            e.printStackTrace();
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    public void registra(View v) {
        if(!precaricato) {
            Toast.makeText(this, "Start recording", Toast.LENGTH_SHORT).show();
            recorder = new MediaRecorder();
            recorder.setAudioSource(MediaRecorder.AudioSource.MIC);
            recorder.setOutputFormat(MediaRecorder.OutputFormat.THREE_GPP);
            createdTime = new Date();
            fileName = getExternalCacheDir().getAbsolutePath() + File.separator + "Nuova_Registrazione_#" + counterFileName + "_" + createdTime.toString().substring(0, 3) + "_" + createdTime.toString().substring(4, 7) + "_" + createdTime.toString().substring(8, 10) + "_" + createdTime.toString().substring(30, 34) + "_" + createdTime.toString().substring(11, 13) + "x" + createdTime.toString().substring(14, 16) + "x" + createdTime.toString().substring(17, 19) + ".3gp";
            recorder.setOutputFile(fileName);
            recorder.setAudioEncoder(MediaRecorder.AudioEncoder.AAC);
            recorder.setAudioEncodingBitRate(16 * 44100);
            recorder.setAudioSamplingRate(44100);
            try {
                recorder.prepare();
            } catch (IOException e) {
                e.printStackTrace();
            }
            recorder.start();
        } else {
            Toast.makeText(this, "Entra in modalità registrazione!", Toast.LENGTH_SHORT).show();
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    public void stopRegistrazione(View v) throws Exception {
        if(!precaricato) {
            try {
                recorder.stop();
                Toast.makeText(this, "Stopped reg", Toast.LENGTH_SHORT).show();

                setRegistrazione(fileName, counterFileName);

                recorder.release();
                recorder = null;

                refresh();

                if (!wave) {
                    centrale.setImageResource(R.drawable.soundwaves);
                    centrale.setMaxHeight(99);
                    centrale.setMaxWidth(231);
                }

            } catch (Exception e) {
                e.printStackTrace();
                Toast.makeText(this, "No recording in progress!", Toast.LENGTH_SHORT).show();
            }
        } else {
            Toast.makeText(this, "Entra in modalità registrazione!", Toast.LENGTH_SHORT).show();
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
            sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_NORMAL);
            accellerometerFlag = true;
        }
    }

    public void setRegistrazione(String fileName, int counterFileName){
        String fileNameAttuale = null, dataAttuale = null;

        if(!precaricato) {

            if (counterFileName >= 0 && counterFileName <= 9) {
                fileNameAttuale = fileName.substring(58, 80);
                dataAttuale = fileName.substring(81, 105);
            } else if (counterFileName >= 10 && counterFileName <= 99) {
                fileNameAttuale = fileName.substring(58, 81);
                dataAttuale = fileName.substring(81, 106);
            } else if (counterFileName >= 100 && counterFileName <= 999) {
                fileNameAttuale = fileName.substring(58, 82);
                dataAttuale = fileName.substring(81, 107);
            } else if (counterFileName >= 1000 && counterFileName <= 9999) {
                fileNameAttuale = fileName.substring(58, 83);
                dataAttuale = fileName.substring(81, 108);
            }

            String fileNameFixed = fileNameAttuale.replaceAll("_", " ");
            String dataAttualeFixed = dataAttuale.replaceAll("x", ":").replaceAll("_", " ");

            nomeTraccia.setText(fileNameFixed);
            dataTraccia.setText(dataAttualeFixed);

        } else {
            int posizioneParola = posizione/5;
            fileNameAttuale = speech[posizioneParola];
            risorsa = speechUnit[posizione];
            nomeTraccia.setText(fileNameAttuale);
            dataTraccia.setText(vocal[posizione%5]);
        }
    }

    public void skipRight(View view) {
        if(!precaricato) {
            try {
                adj--;
                fileName = fileNames.get(counterFileName - adj).toString();
                setRegistrazione(fileName, counterFileName);
                fileName = fileNames.get(counterFileName - adj).getAbsolutePath();
            } catch (Exception e) {
                e.printStackTrace();
                Toast.makeText(this, "No more audio files", Toast.LENGTH_SHORT).show();
                adj++;
            }
        } else {
            try {
                posizione++;
                setRegistrazione(null, 0);
                editor.putInt("pos", posizione);
                editor.commit();
            } catch (Exception e) {
                e.printStackTrace();
                Toast.makeText(this, "No more audio files", Toast.LENGTH_SHORT).show();
                posizione--;
            }
        }
    }

    public void skipLeft(View view) {
        if(!precaricato) {
            try {
                adj++;
                fileName = fileNames.get(counterFileName - adj).toString();
                setRegistrazione(fileName, counterFileName);
                fileName = fileNames.get(counterFileName - adj).getAbsolutePath();
            } catch (Exception e) {
                e.printStackTrace();
                Toast.makeText(this, "No more audio files", Toast.LENGTH_SHORT).show();
                adj--;
            }
        } else {
            try {
                posizione--;
                setRegistrazione(null, 0);
                editor.putInt("pos", posizione);
                editor.commit();
            } catch(Exception e) {
                e.printStackTrace();
                Toast.makeText(this, "No more audio files", Toast.LENGTH_SHORT).show();
                posizione++;
            }
        }
    }

    public void delete(View view) {
        if(!precaricato) {
            try {
                if(fileNames.size() == 0){
                    nomeTraccia.setText("Nessuna traccia audio");
                    dataTraccia.setText(" ");
                    centrale.setImageResource(R.drawable.privacy);
                    centrale.setMaxHeight(99);
                    centrale.setMaxWidth(231);
                    Toast.makeText(this, "No more recording left", Toast.LENGTH_SHORT).show();
                }

                fileNames.get(counterFileName - adj).delete();
                System.out.println(counterFileName - adj);
                System.out.println(fileNames.get(counterFileName - adj).toString());
                System.out.println(fileNames.size());
                refresh();
                setRegistrazione(fileNames.get(counterFileName - adj).toString(), fileNames.size());
            } catch (IndexOutOfBoundsException e) {
                e.printStackTrace();
            }
        } else {
            Toast.makeText(this, "Non puoi cancellare le registrazioni preregistrate!", Toast.LENGTH_SHORT).show();
        }
    }

    private void refresh(){
        if(!precaricato) {
            fileNames = new ArrayList<File>();
            directory = getExternalCacheDir();

            for (File file : directory.listFiles())
                fileNames.add(file);

            counterFileName = directory.listFiles().length;
        }
    }


    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater menuINF = getMenuInflater();
        menuINF.inflate(R.menu.menu, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case R.id.menu_setter: showSetter(); break;
            case R.id.menu_switch: if(item.getTitle().toString().equals("Switch to registered audio")){
                                        precaricato = true;
                                        item.setTitle("Switch to recording audio");
                                        skipLeft(null);
                                        skipRight(null);
                                   } else if(item.getTitle().toString().equals("Switch to recording audio")) {
                                       precaricato = false;
                                       item.setTitle("Switch to registered audio");
                                       skipRight(null);
                                       skipLeft(null);
                                   } break;
        }
        return false;
    }

    public void showSetter(){
        AlertDialog.Builder alert = new AlertDialog.Builder(this);

        LayoutInflater inflater = getLayoutInflater();

        View view = inflater.inflate(R.layout.remote_panel, null);

        ipEdit = (EditText) view.findViewById(R.id.editTextIp);
        portEdit = (EditText) view.findViewById(R.id.editTextPort);
        setButton = (Button) view.findViewById(R.id.buttonSet);

        ipEdit.setText(ip);
        portEdit.setText(port);

        alert.setView(view);

        AlertDialog dialog = alert.create();

        dialog.getWindow().requestFeature(Window.FEATURE_NO_TITLE);
        dialog.show();

        setButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                ip = ipEdit.getText().toString();
                port = portEdit.getText().toString();

                editor.putString("IP", ip);
                editor.putString("PORT", port);
                editor.commit();
            }
        });
    }

    class TaskBackground extends AsyncTask<Integer, Integer, Integer> {

        @Override
        protected Integer doInBackground(Integer... integers) {
            while(accellerometerFlag) {
                 if(audioChecker.isMusicActive()) {
                     accellerometerFlagSeconds = true;
                     try {
                         stream = openFileOutput("data" + (counterFileName - adj) + "_" + personalCounter + ".csv", Context.MODE_PRIVATE);
                         String scrivi = "TIME," + "X," + "Y," + "Z\n";
                         stream.write(scrivi.getBytes());
                     } catch (Exception e) {
                         e.printStackTrace();
                     }
                 }
            }
            return 0;
        }
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (accellerometerFlagSeconds) {
            t = event.timestamp;
            accellerometerFlagSeconds = false;
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
    }
}

