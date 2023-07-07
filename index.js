const qry = (...query) => document.querySelector(...query);
const qrys = (...query) => document.querySelectorAll(...query);


const workspace = qry("#workspace");


// while the conf change, modify workspace accordingly
const workspaceConf = new Proxy({ movementX: 0, movementY: 0, scale: 1 }, {
	set: (target, key, value) => {
		target[key] = value;


		if (key == "scale") {
			// apply scale
			workspace.style["scale"] = `${target.scale}`;
		}


		if (["showingX", "showingY"].includes(key)) {
			// calc the actually movement
			target.movementX = -target.showingX * target.scale;
			target.movementY = -target.showingY * target.scale;
		}


		if (["movementX", "movementY", "scale"].includes(key)) {
			// recalc the showing point
			target.showingX = -target.movementX / target.scale;
			target.showingY = -target.movementY / target.scale;
		}


		if (["movementX", "movementY", "showingX", "showingY"]) { // i.e., if movement changed
			workspace.style["translate"] = `${target.movementX}px ${target.movementY}px`;
			// note: the "translate"d amounts seem not to be affected by "scale"
		}
	}
});


// init workspace
for (let key in workspaceConf) {
	workspaceConf[key] = workspaceConf[key];
}


// move the workspace while mouse dragging
qry("#workspaceContainer").addEventListener("mousedown", e => {
	if (e.target != qry("#workspaceContainer")) return;
	// prevent unexpected workspace movement (e.g. while blocks being dragged)


	const move = e => { // mousemove event
		workspaceConf.movementX += e.movementX;
		workspaceConf.movementY += e.movementY;
	}


	document.body.addEventListener("mousemove", move);
	document.body.addEventListener(
		"mouseup",
		e => document.body.removeEventListener("mousemove", move), { once: true }
	);
});


// scale the workspace while scrolling
{
	let mousePosInWsC = [0, 0]; // mouse's position in workspace container
	qry("#workspaceContainer").addEventListener("mousemove", e => {
		let wsCBox = workspaceContainer.getBoundingClientRect();
		mousePosInWsC = [e.x - wsCBox.x, e.y - wsCBox.y];
		// we can't just set that to [e.layerX, e.layerY] 'cuz
		// the "layer" may be any block the cursor's hovering on
	});


	qry("#workspaceContainer").addEventListener("wheel", e => {
		let oldScale = workspaceConf.scale;


		workspaceConf.scale += e.deltaY * -0.001;
		let newScale = workspaceConf.scale = Math.max(0.05, workspaceConf.scale);


		// use mouse's pos as the scaling center
		let vec = [
			mousePosInWsC[0] - workspaceConf.movementX,
			mousePosInWsC[1] - workspaceConf.movementY
		];


		let vecScale = (1 - newScale / oldScale);
		workspaceConf.movementX += vec[0] * vecScale;
		workspaceConf.movementY += vec[1] * vecScale;
	}, { passive: true });
}


function createBlock({
	header,
	content,
	position: [x = workspaceConf.showingX, y = workspaceConf.showingY] = []
}) {
	// todo: change this element to import and export to json files
	let block = document.createElement("div");
	block.classList.add("block");
	block.innerHTML = `
<div class="inputPorts"></div>
<div class="header">${header || ""}</div>
<div class="content">${content || ""}</div>
<div class="outputPorts"></div>
`;


	block.style["left"] = `${x}px`;
	block.style["top"] = `${y}px`;


	block.children[1].addEventListener("mousedown", e => {
		const move = e => { // mousemove
			x += e.movementX / workspaceConf.scale;
			y += e.movementY / workspaceConf.scale;
			block.style["left"] = `${x}px`;
			block.style["top"] = `${y}px`;
		}


		document.body.addEventListener("mousemove", move);
		document.body.addEventListener(
			"mouseup",
			e => document.body.removeEventListener("mousemove", move), { once: true }
		);
	});


	return block;
}


qry("#addBlock").addEventListener("click", e => {
	qry("#workspace").appendChild(
		createBlock({
			header: prompt("Please write some HTML for the header"),
			content: prompt("Please write some HTML for the content")
		})
	);
});


// create a Input block
qry("#workspace").appendChild(createBlock({ header: "Input" }));


var lineAddStatus = false;


//add lines between blocks by pressing both blocks
function createLines({
	startX,
	startY,
	endX,
	endY
}) {
	//Check Button state
	/*let lineButton = document.getElementById('addLine');
	let buttonState = false;
	lineButton.addEventListener('click', function handleClick() {
	if (!buttonState) buttonState = true;
	else buttonState = false;
	});
	while (buttonState) { */


	let line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
	line.setAttribute('id', 'line2');
	line.setAttribute('x1', startX);
	line.setAttribute('y1', startY);
	line.setAttribute('x2', endX);
	line.setAttribute('y2', endY);
	line.setAttribute("stroke", "black");
	$("svg").append(line);
	//}




}

function selectBlock(){
	qry("#workspace").addEventListener('click', (e) => {
		let clickedElement = e.parentElement;
		while(!e.classList.contains("block")){
			let clickedElement = clickedElement.parentElement; // this line is changed
		} 
		clickedElement.style["background-color"] = "#d3d3d3";
			// strings need "" around
			// e != clickedElement
	}, {once: true}) // this makes it run once
}


qry("#addLine").addEventListener("click", e => {
	if (!lineAddStatus) {
		lineAddStatus = true;
		document.getElementById("addLine").innerHTML = "click two blocks";
	} else {
		lineAddStatus = false;
		document.getElementById("addLine").innerHTML = "Connect Blocks";
	}
})


while (lineAddStatus) {
	selectBlock;
}


