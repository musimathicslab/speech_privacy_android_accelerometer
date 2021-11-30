package com.example.spy;

import android.media.MediaPlayer;
import android.os.Build;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.EditText;
import android.widget.Spinner;

import androidx.annotation.RequiresApi;

import java.io.IOException;

public class AutoPlayActivity {
    Spinner scaletta3;
    /*player2 = MediaPlayer.create(getApplicationContext(), R.raw.silence);
    scaletta3 = findViewById(R.id.scaletta3);
    ArrayAdapter<String> adapter3 =new ArrayAdapter<String>(this,R.layout.element,R.id.elemento, speech);
        scaletta3.setAdapter(adapter3);
        scaletta3.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
        @Override
        public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {

            unit=parent.getSelectedItemPosition()*5;

            risorseAutoPlay[0]=speechUnit[unit];
            risorseAutoPlay[1]=speechUnit[unit+1];
            risorseAutoPlay[2]=speechUnit[unit+2];
            risorseAutoPlay[3]=speechUnit[unit+3];
            risorseAutoPlay[4]=speechUnit[unit+4];

        }

        @Override
        public void onNothingSelected(AdapterView<?> parent) {

        }
    });

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    public void autoPlay(View v) throws InterruptedException, IOException {
        int contResource = 0;
        int contTime = 0;
        boolean fine=true;
        int times=1;
        EditText editText=findViewById(R.id.repeatTime);
        if(editText!=null && editText.getText()!=null && !editText.getText().toString().equals("")){
            times=Integer.parseInt(editText.getText().toString());
        }
        player = MediaPlayer.create(getApplicationContext(), risorseAutoPlay[contResource]);
        background(findViewById(R.id.buttonBack));
        while (contResource < risorseAutoPlay.length) {
            fine=false;
            contTime=0;
            while (fine==false) {
                if (!player.isPlaying() && !player2.isPlaying()) {
                    if (contTime == times) {
                        contResource++;
                        if(contResource<risorseAutoPlay.length) {
                            player = MediaPlayer.create(getApplicationContext(), risorseAutoPlay[contResource]);
                        }
                        fine=true;
                    }else{
                        riproduciExample(v);
                        contTime++;
                    }
                } else {
                    //wait
                }
            }
        }
        background(findViewById(R.id.buttonBack));
    }*/
}
