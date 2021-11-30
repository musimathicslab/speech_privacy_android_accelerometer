package com.example.spy;

import android.Manifest;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.media.AudioManager;
import android.media.MediaPlayer;
import android.media.MediaRecorder;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.Spinner;
import android.widget.TextView;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.widget.ListViewCompat;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

import static android.os.Process.THREAD_PRIORITY_BACKGROUND;


public class MainActivity extends AppCompatActivity implements SensorEventListener , AdapterView.OnItemSelectedListener  {

    File path = null;
    ImageButton rec = null;
    ImageButton stop = null;
    MediaRecorder recorder = null;
    private MediaPlayer player = null;
    private MediaPlayer player2 = null;
    boolean registrazione = false;
    String fileName = null;
    int cont = 0;
    boolean flag = true;
    long t;
    FileOutputStream stream;
    AudioManager am;
    SensorManager manager;
    Sensor accelerometer;
    boolean flagStart = false;
    boolean flagBackground=false;
    SensorEventListener listener = this;
    int contFile = 1;
    SharedPreferences preferences;
    TextView statoReg;
    ImageButton registraBack;
    Spinner spinner;
    String selection;
    Spinner scaletta;
    int risorsa;
    Spinner scaletta2;
    int[] speechUnit={R.raw.a1,R.raw.a2,R.raw.a3,R.raw.a4,R.raw.a5,R.raw.all1,R.raw.all2,R.raw.all3,
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
    String[] vocal = {"A","B","C","D","E"};
    int[] risorse=new int[5];
    int unit=0;
    TextView responseTextTimer;


    @RequiresApi(api = Build.VERSION_CODES.M)
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        preferences = getSharedPreferences("contatore", MODE_PRIVATE);
        contFile = preferences.getInt("cont", 1);
        manager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        accelerometer = manager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        path = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS);
        requestPermissions(new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.RECORD_AUDIO,Manifest.permission.INTERNET}, PackageManager.PERMISSION_GRANTED);
        rec = findViewById(R.id.rec);
        stop = findViewById(R.id.stopRec);
        fileName = getExternalCacheDir().getAbsolutePath();
        fileName += "/audiorecordtest.3gp";
        statoReg =findViewById(R.id.statoReg);
        registraBack=findViewById(R.id.buttonBack);
        responseTextTimer = findViewById(R.id.responseText);
        am = (AudioManager) getSystemService(Context.AUDIO_SERVICE);
        risorsa =R.raw.a1;
        player = MediaPlayer.create(getApplicationContext(), risorsa);
        spinner = findViewById(R.id.spinner);
        setSpinnerAdapter();
        spinner.setOnItemSelectedListener(this);
        scaletta = findViewById(R.id.scaletta);
        ArrayAdapter<String> adapter1 =new ArrayAdapter<String>(this,R.layout.element,R.id.elemento, speech);
        scaletta.setAdapter(adapter1);
        scaletta.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {

                unit=parent.getSelectedItemPosition()*5;
                player = MediaPlayer.create(getApplicationContext(), speechUnit[unit]);

                    risorse[0]=speechUnit[unit];
                    risorse[1]=speechUnit[unit+1];
                    risorse[2]=speechUnit[unit+2];
                    risorse[3]=speechUnit[unit+3];
                    risorse[4]=speechUnit[unit+4];

            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {

            }
        });
        scaletta2=findViewById(R.id.scaletta2);
        ArrayAdapter<String> adapter2 =new ArrayAdapter<String>(this,R.layout.element,R.id.elemento, vocal);
        scaletta2.setAdapter(adapter2);
        scaletta2.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                String sel = parent.getSelectedItem().toString();
                if (sel.equals("A")) {
                    risorsa = risorse[0];
                    player = MediaPlayer.create(getApplicationContext(), risorsa);
                }else if(sel.equals("B")) {
                    risorsa = risorse[1];
                    player = MediaPlayer.create(getApplicationContext(), risorsa);
                }else if(sel.equals("C")) {
                    risorsa = risorse[2];
                    player = MediaPlayer.create(getApplicationContext(), risorsa);
                }else if(sel.equals("D")) {
                    risorsa = risorse[3];
                    player = MediaPlayer.create(getApplicationContext(), risorsa);
                }else if(sel.equals("E")) {
                    risorsa = risorse[4];
                    player = MediaPlayer.create(getApplicationContext(), risorsa);
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {

            }
        });

        rec.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (!registrazione) {
                    registra(v);
                } else {
                    stopRegistrazione(v);
                }
                registrazione = !registrazione;
            }
        });
        stop.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                riproduci(v);
            }
        });
    }



    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    public void registra(View v) {
        ((ImageButton)v).setBackground(getDrawable(R.drawable.ic_baseline_pause_24));
        recorder = new MediaRecorder();
        recorder.setAudioSource(MediaRecorder.AudioSource.MIC);
        recorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
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
    public void stopRegistrazione(View v) {
        ((ImageButton)v).setBackground(getDrawable(R.drawable.ic_baseline_mic_24));
        recorder.stop();
        recorder.release();
        recorder = null;

    }

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


    @Override
    public void onSensorChanged(SensorEvent event) {
        if (flag) {
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

    }

    public void salvaContFile() {
        SharedPreferences.Editor edit = preferences.edit();
        edit.putInt("cont", contFile);
        edit.commit();
    }


    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
    }

    @Override
    public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
        selection=parent.getSelectedItem().toString();

    }

    @Override
    public void onNothingSelected(AdapterView<?> parent) {

    }

    class TaskBackground extends AsyncTask<Integer, Integer, Integer> {

        @Override
        protected Integer doInBackground(Integer... integers) {

            while (flagBackground) {
                if (am.isMusicActive()&&!flagStart) {
                    flag = true;
                    flagStart = true;
                    statoReg.post(new Runnable() {
                        @Override
                        public void run() {
                            statoReg.setText(" ON");
                        }
                    });
                    manager.registerListener(listener, accelerometer, SensorManager.SENSOR_DELAY_FASTEST);
                    try {
                        stream = openFileOutput("data" + contFile + ".csv", Context.MODE_PRIVATE);
                        String scrivi = "TIME," + "X," + "Y," + "Z\n";
                        stream.write(scrivi.getBytes());
                        contFile++;
                        salvaContFile();
                    } catch (FileNotFoundException e) {
                        e.printStackTrace();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                if (!am.isMusicActive()&&flagStart) {
                    statoReg.post(new Runnable() {
                        @Override
                        public void run() {
                            statoReg.setText(" OFF");
                        }
                    });
                    flagStart = false;
                    manager.unregisterListener(listener);
                    spinner.post(new Runnable() {
                        @Override
                        public void run() {
                            setSpinnerAdapter();
                        }
                    });
                    try {
                        stream.close();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }

            }

            return 0;

        }
    }

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    public void background(View v) {

        if(!flagBackground)
        {
            ((ImageButton)v).setBackground(getDrawable(R.drawable.ic_baseline_pause_24));
            flagBackground=true;
            new TaskBackground().execute(0);

        }
        else
        {
            flagBackground=false;
            ((ImageButton)v).setBackground(getDrawable(R.drawable.ic_baseline_mic_24));
        }
    }



    public void setSpinnerAdapter()
    {
        int n= preferences.getInt("cont", 1);
        ArrayList<String> file =new ArrayList<String>();
        for(int i=1; i<n; i++)
        {
            file.add("data"+i+".csv");
        }
        TextView responseText = findViewById(R.id.countFileText);
        responseText.setText("Repeat: "+(n-1));
        ArrayAdapter<String> adapter =new ArrayAdapter<String>(this,R.layout.element,R.id.elemento,file);
        spinner.setAdapter(adapter);
    }

    public void invia(View v) throws IOException {
        EditText ipv4AddressView = findViewById(R.id.ip);
        String ipv4Address = ipv4AddressView.getText().toString();
        EditText portNumberView = findViewById(R.id.port);
        String portNumber = portNumberView.getText().toString();


        String postUrl= "http://"+ipv4Address+":"+portNumber+"/";

        String path = getFilesDir().getAbsolutePath()+"/"+selection;
        File file = new File(path);
        RequestBody postBodyImage = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("image", selection, RequestBody.create(MediaType.parse("text/csv"), file))
                .build();
        if(ipv4Address!=null && portNumber!=null && postBodyImage!=null &&
                !ipv4Address.equals("") && !portNumber.equals("")) {
            ipv4AddressView.setBackgroundColor(R.drawable.border);
            portNumberView.setBackgroundColor(R.drawable.border);
            spinner.setBackgroundColor(R.drawable.border);
            postRequest(postUrl, postBodyImage);
        }else if(ipv4Address==null||ipv4Address.equals("")){
            ipv4AddressView.setBackgroundColor(Color.RED);
        }else if(portNumber==null||portNumber.equals("")){
            portNumberView.setBackgroundColor(Color.RED);
        }else if(postBodyImage==null){
            spinner.setBackgroundColor(Color.RED);
        }
    }

    void postRequest(String postUrl, RequestBody postBody) {

        OkHttpClient client = new OkHttpClient();

        Request request = new Request.Builder()
                .url(postUrl)
                .post(postBody)
                .build();

        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                // Cancel the post on failure.
                call.cancel();

                // In order to access the TextView inside the UI thread, the code is executed inside runOnUiThread()
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        TextView responseText = findViewById(R.id.responseText);
                        responseText.setText("Failed to Connect to Server ->"+postUrl);
                    }
                });
            }

            @Override
            public void onResponse(Call call, final Response response) throws IOException {
                // In order to access the TextView inside the UI thread, the code is executed inside runOnUiThread()
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        TextView responseText = findViewById(R.id.responseText);
                        try {
                            String r=response.body().string();
                            r=r.substring(1,r.length()-1);
                            int risp=Integer.valueOf(r);
                            responseText.setText("File SALVATO IN MEMORIA \n SPEECH PRONUNCIATA: "+speech[risp]);
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    }
                });
            }
        });
    }


    public void riproduciExample(View v) throws IOException {
        if(!player.isPlaying() && risorsa!=0) {
            player.start();
        }

    }
    //end
}


