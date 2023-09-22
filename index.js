
//I don't know how to avoid using all the function in global scope. all items with "window." prefix should be fixed in some way.

//initailize ans util------------------------------------------------------------------------------------------------------------------
var idMax = -1;
function getIdNumber(string) {//get id number from any form of id string whose id number is its suffix
    var idNum=0;
    for(let i = string.length-1, k = 1;string[i]!=':';i--) {
        idNum+=k*parseInt(string[i]);
        k*=10;
    }
    return idNum;
}
function test() { //just to test whether a event is triggered correctly
    console.log("test");
}
window.test = test;
var blockDict = 
//initailize-----------------------------------------------------------------------------------------------------------------------------

//drag and drop--------------------------------------------------------------------------------------------------------------------------

function drag_start(event) {
    var style = window.getComputedStyle(event.target, null);
    var idNum = getIdNumber(event.target.id);
    event.dataTransfer.setData("text/plain",
    (parseInt(style.getPropertyValue("left"),10) - event.clientX) + ',' + (parseInt(style.getPropertyValue("top"),10) - event.clientY)+','+idNum);
    return false;
} 
window.drag_start = drag_start;

function drag_over(event) { 
    event.preventDefault(); 
    return false; 
} 
window.drag_over = drag_over;

function drop(event) { 
    var offset = event.dataTransfer.getData("text/plain").split(',');
    var dm = document.getElementById('blockID:'+offset[2]);
    dm.style.left = (event.clientX + parseInt(offset[0],10)) + 'px';
    dm.style.top = (event.clientY + parseInt(offset[1],10)) + 'px';
    event.preventDefault();
    return false;
} 
window.drop = drop;

//drag and drop--------------------------------------------------------------------------------------------------------------------------
//dbclick for side collapse bar------------------------------------------------------------------------------------------------------------
function openParameterBarve(event) {
    var idNum = getIdNumber(event.target.id);
    document.getElementById("PBarID:" + idNum).style.width = "250px";
    //document.getElementById("main").style.marginLeft = "250px";
}
window.openParameterBar = openParameterBar;

function closeParameterBar(event) {
    var idNum = getIdNumber(event.target.id);
    document.getElementById("PBarID:" + idNum).style.width = "0";
    //document.getElementById("main").style.marginLeft= "0";
}
window.closeParameterBar = closeParameterBar;
//dbclick for side collapse bar------------------------------------------------------------------------------------------------------------

//add a new block and parameter bar----------------------------------------------------------------------------------------------------------------------------
function addBlock() {
    idMax++;
    const ele = document.createElement("div"); 
    ele.setAttribute('id', 'blockID:' + idMax);
    ele.setAttribute('class', 'block');
    ele.setAttribute('draggable', true);
    ele.setAttribute('ondragstart', "drag_start(event)");

    const type = document.getElementById('addBlock-modal-selecter').value;
    console.log(type);
    ele.setAttribute('blockType', type);
    const text = document.createElement("h2");
    text.textContent = type;
    ele.appendChild(text);
    const addBlcokcloseButton = document.createElement("a");
    addBlcokcloseButton.setAttribute('href', 'javascript:void(0)');
    addBlcokcloseButton.setAttribute('id', 'addBlockCloseButton:' + idMax);
    addBlcokcloseButton.setAttribute('onclick','deleteBlock(event)');
    addBlcokcloseButton.textContent = "X";
    ele.appendChild(addBlcokcloseButton);

    const workspace = document.getElementById('workSpace');
    workspace.appendChild(ele);

    addParaBar(idMax);
}
window.addBlock = addBlock;

function addParaBar(idMax) {
    const ele = document.createElement("div");
    const paraCloseButton = document.createElement("a");
    paraCloseButton.setAttribute('href', 'javascript:void(0)');
    paraCloseButton.setAttribute('id', 'PBarID:' + idMax);
    paraCloseButton.setAttribute('class', 'parameterBar');
    paraCloseButton.setAttribute('onclick','closeParameterBar(event)');
}
window.addParaBar = addParaBar;
//add a new block and parameter bar----------------------------------------------------------------------------------------------------------------------------

//delete a block and paarmeter bar----------------------------------------------------------------------------------------------------------------------------
function deleteBlock(event) {
    var idNum = getIdNumber(event.target.id);
    console.log(idNum);
    
    document.getElementById('blockID:' + idNum).remove();
    for(let i = idNum+1;i <= idMax; i++) {
        let curBlk = document.getElementById('blockID:' + i);
        curBlk.setAttribute('id', 'blockID:' + (i-1));
        let curClsB = document.getElementById('addBlockCloseButton:' + i);
        curClsB.setAttribute('id', 'addBlockCloseButton:' + (i-1));

    }
    idMax--;
    console.log(idMax);
    
}
window.deleteBlock = deleteBlock;

//delete block----------------------------------------------------------------------------------------------------------------------------
