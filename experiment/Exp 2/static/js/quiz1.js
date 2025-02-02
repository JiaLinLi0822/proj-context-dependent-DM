var quiz1choice=new Array();
quiz1choice[0]=new Array(-2.5,-4.5);
quiz1choice[1]=new Array(-1.9,-3.3);
quiz1choice[2]=new Array(-4.9,-2.2);
var correctchoice=new Array("1","2","2");
var correctdisplay=new Array(-2.5,-3.3,-2.2);
var wrongdisplay=new Array(-4.5,-1.9,-4.9);
var num_quiz1=1;
var num_correct=0;
var num_wrong=0;
var correctorwrong;
var quiz1data=new Array();

$(document).ready(function(){
	$('#next').hide();
	$('#quiz1continue').hide();
	$('#quiz1button').click(function(){
		var subjchoice = $("input[name='radioname']:checked").val();
		var check_or_not=$('input:radio[name="radioname"]').is(":checked");
		//psiTurk.recordTrialData({"phase":"QUIZ","quiz_num":1,"trytime":1,"action":"SubmitAnswer","question":num_quiz1 , "checkornot":check_or_not, "answer":subjchoice});
		//psiTurk.saveData();
		var trialdata=new Array(1,1,num_quiz1,Number(check_or_not),subjchoice,Number(subjchoice===correctchoice[num_quiz1-1]));
		quiz1data.push(trialdata);
		feedback(subjchoice,check_or_not);
	})
	$('#quiz1continue').click(function(){
		clickcontinue();
	})
	$('#next').click(function(){
		clicknext();
	})
});

function feedback(subjchoice,check_or_not){
	if(check_or_not){
		document.getElementById("quiz1button").disabled = true;
		feedback_correct1="回答正确！让我们再做另外一道。";
		feedback_wrong1="抱歉，答案并不正确。评分为 "+correctdisplay[num_quiz1-1].toString()+" 的bar要比评分为 " +wrongdisplay[num_quiz1-1].toString()+" 的bar更高, 因此评分为 "+correctdisplay[num_quiz1-1].toString()+" 的餐馆数量更多。让我们再试一次";
		feedback_correct2="回答正确！";
		feedback_wrong2="抱歉，答案并不正确。评分为 "+correctdisplay[num_quiz1-1].toString()+" 的bar要比评分为 " +wrongdisplay[num_quiz1-1].toString()+" 的bar更高, 因此评分为 "+correctdisplay[num_quiz1-1].toString()+" 的餐馆数量更多。";
		
		if(subjchoice === correctchoice[num_quiz1-1]){correctorwrong=1;num_correct++;}
		else{correctorwrong=0;num_wrong++;}
		
		if(num_wrong>=2){
			$('#quiz1continue').show();
			if(correctorwrong===1){$('#quiz1feedback').text(feedback_correct2);}
			else{$('#quiz1feedback').text(feedback_wrong2);}
		}
		else if(num_correct>=2){
			if(correctorwrong===1){$('#quiz1feedback').text(feedback_correct2);$('#next').show();}
			else{$('#quiz1feedback').text(feedback_wrong2);$('#next').show();}
		}
			else{
				$('#quiz1continue').show();
				if(correctorwrong===1){$('#quiz1feedback').text(feedback_correct1);}
				else{$('#quiz1feedback').text(feedback_wrong1);}
			}
	}
	else{
		$('#quiz1feedback').text("请再选择答案后提交。");
	}

}

function clickcontinue(){
	if(num_quiz1<3&&num_wrong<2){
		num_quiz1++;
		$('input[name=radioname]').attr('checked',false);
		$('#quiz1feedback').text([]);
		$('#quiz1continue').hide();
		for(var i=1;i<=2;i++){
			var choiceid='#quiz1choice'+i.toString();
			$(choiceid).text(quiz1choice[num_quiz1-1][i-1].toString());
		}
		document.getElementById("quiz1button").disabled = false;
	}
	else{
		$('#quiz1data').text(quiz1data);
		window.open('not_qualified.html', '_blank');
	}
}

function clicknext(){
	$('#quiz1data').text(quiz1data);
	window.open('instruction3_result.html', '_blank');
}