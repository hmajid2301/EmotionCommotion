function UpdateEmoji() {

	index = GetEmotion()

	$("#angry,#happy,#sad,#neutral").css({ width: "0px" });
	switch (index) {

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