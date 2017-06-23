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
var isRecording = false;

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
    returnedFrames = 0;


    //hide microphone
    //show waveform and stop button
    //start recording
    //set interval to loop, i.e. keep recording audio data
    $("#microphone").hide()
    $("#emojis").hide()
    $("#waveform").show()
    $("#stop").show()
    toggleRecording(this)
    microphone.start()
    interval = setInterval(loop, 2000)
    isRecording = true;
});


//on click of stop button
$("#stop").click(function () {

    //clear interval to stop loop
    //show microphone
    //hide waveform and stop and graph
    //end recording
    //restart frame number
    clearInterval(interval);
    $("#microphone").show()
    $("#waveform").hide()
    $("#stop").hide()
    toggleRecording(document.getElementById("microphone"))
    microphone.stop()
    frameNum = 0;
    var frameCaught = true;
    isRecording = false;

    //when all ajax calls have returned
    //show the dominant emoji
    $(document).ajaxStop(function () {
        $(document).off('ajaxStop')
        $("#graph").hide()
        loadEmoji(emotion);
        frameCaught = false;
    });

    //if a frame gets sent sent to server, but user
    //ends recording then ajax call will never receive
    //so show emoji from current set
    if (frameCaught) {
        $(document).off('ajaxStop')
        $("#graph").hide()
        setTimeout(loadEmoji, 2000, emotion);
    }
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

        //ajax call
        //POST to blob
        $.ajax({
            url: "/blob",
            type: "post",
            data: data,
            processData: false,
            contentType: false,
            success: function (a) {
              if(isRecording) {
                callback(a);
              }
            },
            error: function (e) {
                console.log(e)
            }
        });

    }
    frameNum++;
}

//on successful return of ajax call update emotions, with an averaging function for each emoji
function callback(a) {
    //increment number of returned frames
    returnedFrames++;
    emotion.neu =  Math.floor((parseFloat(a.neu)*100 + emotion.neu * (returnedFrames - 1)) / returnedFrames);
    emotion.hap = Math.floor((parseFloat(a.hap)*100 + emotion.hap * (returnedFrames - 1)) / returnedFrames);
    emotion.ang = Math.floor((parseFloat(a.ang)*100 + emotion.ang * (returnedFrames - 1)) / returnedFrames);
    emotion.sad = Math.floor((parseFloat(a.sad)*100 + emotion.sad * (returnedFrames - 1)) / returnedFrames);
    //show graph on success
    console.log(returnedFrames);
    $("#graph").show()

}

//loop calls doneEncoding but first get blob data
function loop() {
    audioRecorder.getBuffers(gotBuffers);
}

//calculates dominant emoji from dictionary
function loadEmoji(data) {
    var max = 0;
    var dominant;

    for (var key in data) {

        //get probability, if greater than previous highest
        //update highest and store emoji name
        if (max < data[key]) {
            max = data[key];
            dominant = key;
        }
    }

    //show dominant emoji and reset emotion dictionary
    updateEmoji(dominant)
    emotion = {hap: 0, neu: 0, sad: 0,ang: 0};
}


//show dominant emoji hide all the others
function updateEmoji(emot) {


    $("#emojis").show()
    $("#angry,#happy,#sad,#neutral").css({ width: "0px" });
    switch (emot) {

        case 'ang':
            $("#angry").animate({
                width: "30%"
            });
            break;
        case 'sad':
            $("#sad").animate({
                width: "30%"
            });
            break;
        case 'neu':
            $("#neutral").animate({
                width: "30%"
            });
            break;
        case 'hap':
            $("#happy").animate({
                width: "30%"
            });
            break;
    }

}
