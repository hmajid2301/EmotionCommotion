function UpdateEmoji() {
	var max = 100;
	var angry = randombetween(1, max - 3);
	var sad = randombetween(1, max - 2 - angry);
	var neutral = randombetween(1, max - 1 - angry - sad);
	var happy = max - angry - sad - neutral;

	emojVals = [angry, sad, neutral, happy];
	largestIndex = indexOfMaxValue = emojVals.reduce((iMax, x, i, arr) => x > arr[iMax] ? i : iMax, 0);

	$("#angry,#happy,#sad,#neutral").css({width: "0px"});
	switch (largestIndex) {

		case 0:
		    $("angry").fadeIn(1);
		    $("#angry").animate({
		        width: "300px"
		    });
			break;
		case 1:
		    $("#sad").animate({
		        width: "300px"
		    });
			break;
		case 2:
		    $("#neutral").animate({
		        width: "300px"
		    });
			break;
		case 3:
		    $("#happy").animate({
		        width: "300px"
		    });
			break;

	}

}

function randombetween(min, max) {
	return Math.floor(Math.random() * (max - min + 1) + min);
}

(function ($) {
	$.each(['show', 'hide'], function (i, ev) {
		var el = $.fn[ev];
		$.fn[ev] = function () {
			this.trigger(ev);
			return el.apply(this, arguments);
		};
	});
})(jQuery);

$('#emojis').on('show', UpdateEmoji);