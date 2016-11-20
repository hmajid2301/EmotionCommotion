function UpdateEmoji() {

	index = GetEmotion()

	$("#angry,#happy,#sad,#neutral").css({width: "0px"});
	switch (index) {

		case 0:
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