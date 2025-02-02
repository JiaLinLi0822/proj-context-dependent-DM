var questionnum=1;
var trytime=1;
var correct_or_wrong;
var correctanswer=["2","1","1"];
var correctanswerdisplay=["-2.3","-2.5"];
var quiz2data=new Array();

$(document).ready(function(){
	$('#next').hide();
	$('#quiz2continue').hide();
	$('#quiz2button').click(function(){
		var choice = $("input[name='radioname']:checked").val();
		var check=$('input:radio[name="radioname"]').is(":checked");
		//psiTurk.recordTrialData({"phase":"QUIZ", "quiz_num":2,"trytime":trytime, "action":"SubmitAnswer","question":questionnum , "checkornot":check, "answer":choice});
		//psiTurk.saveData();
		var trialdata=new Array(2,questionnum,trytime,Number(check),choice,Number(choice===correctanswer[questionnum-1]));
		quiz2data.push(trialdata);
		if(check){
			$('#quiz2button').attr('disabled',"true");
			if(questionnum<=2){$('#quiz2continue').show();}
			else{$('#next').show();}
			if(choice===correctanswer[questionnum-1]){
				$('#quiz2feedback').text("回答正确！");
				correct_or_wrong=1;
			}
			else{
				if(questionnum===1){
					if(trytime===1){$('#quiz2feedback').text("抱歉，正确答案是 "+correctanswerdisplay[trytime-1]+"。让我们再试一次。");}
					else{$('#quiz2feedback').text("很抱歉，正确答案是 "+correctanswerdisplay[trytime-1]+"。");}
					}
				else if(questionnum===2){
						if(trytime===1){$('#quiz2feedback').text("抱歉，答案并不正确，让我们再试一次。");}
						else{$('#quiz2feedback').text("抱歉，答案并不正确。");}
					}
					else{$('#quiz2feedback').text("正确的答案是“是”。因为这个城市有很多餐馆，当你点击“去一个随机的新餐馆”时，你可能会去到和以前评分相同的餐馆。当你继续的时候，请记住这一点。");}
				correct_or_wrong=0;
			}
		}
		else{
			$('#quiz2feedback').text("请再选择答案后提交。");
		}
	})

	$('#quiz2continue').click(function(){
		$('#quiz2button').removeAttr('disabled');
		if(questionnum<=2){
			if(correct_or_wrong===1){
				if(trytime===2){
					//$('#quiz2img').attr("src","../img/quiz2-1.png");trytime=1;
				    $('#quiz2img').attr({
				        src: $('#quiz2img').attr('srcnew') 
				        , 'srcnew': $('#quiz2img').attr('src') 
				    });
				    trytime--;
				}
				if(questionnum===1){
					$('#quiz2question').text('2. 如果你点击“去一个随机的新餐馆”，你将');
					$('#choice1').text('得到-5.0到-1.0之间的评分；其中评分在-3分左右的可能性更大。');
					$('#choice2').text('得到-5.0到-1.0之间的评分；每一个得分都将是等可能的。');
					$('#choice3').text('只能得到评分在-2.3分以下的评分。');
				}
				else{
					$('#quiz2question').text('3. 如果你点击“去一个随机的新餐馆”，这个新餐馆有可能和之前的餐馆有相同的评分吗？');
					$('#choice1').text('是');
					$('#choice2').text('否');
					$('#quiz2choice3').hide();
				}
				questionnum++;
			}
			else{
				if(trytime===1){
					//$('#quiz2img').attr("src","../img/quiz2-2.png");
					$('#quiz2img').attr({
				        src: $('#quiz2img').attr('srcnew') 
				        , 'srcnew': $('#quiz2img').attr('src') 
				    });
					if(questionnum===1){
						$('#choice1').text('-2.5');
						$('#choice2').text('-3.5');
					}
					else{
						$('#choice3').text('只能得到评分在-2.5分以下的评分。');
					}
					trytime++;
				}
				else{
					$('#quiz2data').text(quiz2data);
					window.open('not_qualified.html', '_blank');
				}

			}
		}
		$('input[name=radioname]').attr('checked',false);
		$('#quiz2continue').hide();
		$('#quiz2feedback').text("");
	})
	
	$('#next').click(function(){
		$('#quiz2data').text(quiz2data);
		window.open('instruction7.html', '_blank');
	})
});
