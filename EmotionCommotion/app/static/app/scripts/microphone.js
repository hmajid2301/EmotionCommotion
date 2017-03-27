var emotion;
var interval = null;
var frameNum = 0;

var wavesurfer = WaveSurfer.create({
    container: '#waveform',
    waveColor: '#367ee9'
});

var microphone = Object.create(WaveSurfer.Microphone);

microphone.init({
    wavesurfer: wavesurfer
});

function GetEmotion() {
    return emotion;
}


$("#microphone").click(function () {
    console.log("Microphone Clicked")

    $("#microphone").hide()
    $("#emojis").hide()
    $("#waveform").show()
    $("#stop").show()
    toggleRecording(this)
    microphone.start()
    interval = setInterval(loop, 500)

});


$("#stop").click(function () {
    console.log("Stop Clicked")

    clearInterval(interval);
    $("#microphone").show()
    $("#waveform").hide()
    $("#stop").hide()
    toggleRecording(document.getElementById("microphone"))
    microphone.stop()
});

// function doneEncoding(blob) {
//     var data = new FormData();
//     frameNum++;
//
//     data.append("fname", "test.wav");
//     data.append("enctype", "multipart/form-data");
//     data.append("data", blob);
//     data.append("frame-number", frameNum);
//     console.log(frameNum);
//
//
//     $.ajax({
//         url: "/blob",
//         type: "post",
//         data: data,
//         processData: false,
//         contentType: false,
//         success: function (a) {
//             emotion = a.emotion
//             $("#emojis").show()
//         },
//         error: function (e) {
//             console.log(e)
//         }
//     });
// }

// http://stackoverflow.com/questions/15970729/appending-blob-data
var MyBlobBuilder = function() {
  this.parts = [];
}

MyBlobBuilder.prototype.append = function(part) {
  this.parts.push(part);
  this.blob = undefined; // Invalidate the blob
};

MyBlobBuilder.prototype.getBlob = function() {
  if (!this.blob) {
    this.blob = new Blob(this.parts, { type: "audio/wav" });
  }
  return this.blob;
};



var lastblob=0;
function doneEncoding(blob) {
    if (frameNum > 0) {
        var data = new FormData();

        data.append("enctype", "multipart/form-data");
        var myBlobBuilder = new MyBlobBuilder();
        data.append("blob", lastblob);
        console.log("lastblob" + lastblob.size)
        console.log("blob" + blob.size)
        myBlobBuilder.append(lastblob)
        myBlobBuilder.append(blob);
        var frame = myBlobBuilder.getBlob();
        console.log("frame" + frame.size)
        data.append("frame", frame);

        data.append("frame-number", frameNum);
        console.log(frameNum);

        $.ajax({
            url: "/blob",
            type: "post",
            data: data,
            processData: false,
            contentType: false,
            success: function (a) {
                $("#graph").show()
            },
            error: function (e) {
                console.log(e)
            }
        });
    }
    frameNum++;

    lastblob = blob;

}

function loop() {
    console.log("Stopping")
    toggleRecording(document.getElementById("microphone"))
    setTimeout(function () { console.log("waiting");}, 200)
}
