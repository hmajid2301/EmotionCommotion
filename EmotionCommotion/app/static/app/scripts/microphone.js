//*************************************************************************************
// microphone.js
// This file records the audio data, sends it to the server
// receives information from the server and updates the emotion data
//
// emotion - dictionary of emotion data
// interval - causes audio functions to loop until user ends recording
// frameNum - number of frames sent to server
//***********************************************************************************
var emotion = {
    hap: 0,
    neu: 0,
    sad: 0,
    ang: 0
};

var interval = null;
var frameNum = 0;
var returnedFrames = 0;

//create waveform, amplitude against time
var wavesurfer = WaveSurfer.create({
    container: '#waveform',
    waveColor: '#367ee9'
});

//set waveform to use the microphone data on the browser
var microphone = Object.create(WaveSurfer.Microphone);
microphone.init({
    wavesurfer: wavesurfer
});

//return emotion dictionary for graph.js
function getEmotion() {
    return emotion;
}


//on click of microphone
$("#microphone").click(function () {

    //hide microphone
    //show waveform and stop button
    //start recording
    //set interval to loop, i.e. keep recording audio data
    $("#microphone").hide()
    $("#waveform").show()
    $("#stop").show()
 
    toggleRecording(this)
    microphone.start()
    interval = setInterval(loop, 2000)
});


//on click of stop button
$("#stop").click(function () {

    //clear interval to stop loop
    //show microphone
    //hide waveform and stop and graph
    //end recording
    clearInterval(interval);
    $("#microphone").show()
    $("#waveform").hide()
    $("#stop").hide()
    $("#graph").hide()
    toggleRecording(document.getElementById("microphone"))
    microphone.stop()
    frameNum = 0;
});


//done encoding called every 1 second
//ajax call
function doneEncoding(blob) {

    //done send on first call, so we get an overlap
    if (frameNum > 0) {

        //create a form, of data for backend
        var data = new FormData();
        data.append("enctype", "multipart/form-data");
        data.append("blob", blob);
        data.append("frame-number", frameNum);
        console.log("Sending Framenum ", frameNum)
        //ajax call
        //POST to blob
        $.ajax({
            url: "/blob",
            type: "post",
            data: data,
            processData: false,
            contentType: false,
            success: function (a) {
                console.log(a);
                callback(a);
            },
            error: function (e) {
                console.log(e)
            }
        });

    }
    frameNum++;
}

function callback(a) {
    console.log("Major Frame", a)
    returnedFrames++;
    emotion.neu = (parseFloat(a.neu) + emotion.neu * (returnedFrames - 1)) / returnedFrames;
    emotion.hap = (parseFloat(a.hap) + emotion.hap * (returnedFrames - 1)) / returnedFrames;
    emotion.ang = (parseFloat(a.ang) + emotion.ang * (returnedFrames - 1)) / returnedFrames;
    emotion.sad = (parseFloat(a.sad) + emotion.sad * (returnedFrames - 1)) / returnedFrames;

    $("#graph").show()

}

//loop calls doneEncoding but first get blob data
function loop() {
    audioRecorder.getBuffers(gotBuffers);
}
