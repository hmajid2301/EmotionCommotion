function UpdateEmoji() {
	var max = 100;
	var angry = randombetween(1, max - 3);
	var sad = randombetween(1, max - 2 - angry);
	var neutral = randombetween(1, max - 1 - angry - sad);
	var happy = max - angry - sad - neutral;

	emojVals = [angry, sad, neutal, happy];
	largestIndex = arr.emojVals(Math.max.apply(Math, emojVals))

	switch (largestIndex) {
		case 0:
			$("#angry").animate({
				width: "300px"
			})
			break;
		case 1:
			$("#sad").animate({
				width: "300px"
			})
			break;
		case 2:
			$("#neutral").animate({
				width: "300px"
			})
			break;
		case 3:
			$("#happy").animate({
				width: "300px"
			})
			break;

	}



	$("#sad").animate({
		width: sad * 3 + "px"
	})
	$("#neutral").animate({
		width: neutral * 3 + "px"
	})
	$("#happy").animate({
		width: happy * 3 + "px"
	})
}


$('#emoji-toggle').change(function () {
	if ($(this).prop('checked') == true) {
		$("#emojis").show()
		$("#graph").hide()
	}
	else {
		$("#graph").show()
		$("#emojis").hide()
	}
})

function randombetween(min, max) {
	return Math.floor(Math.random() * (max - min + 1) + min);
}

