const qry = (...query) => document.querySelector(...query);
const qrys = (...query) => document.querySelectorAll(...query);

const workspace = qry("#workspace");

// when .x and .y change, modify #workspace accordingly
const workspaceConf = new Proxy({showingX: 0, showingY: 0, scale: 1}, {
	set: (target, key, value) => {
		target[key] = value;

		switch(key){
			case "showingX":
			case "showingY":
				workspace.style["translate"] = `${-target.showingX}px ${-target.showingY}px`;
				break;
			case "scale":
				workspace.style["scale"] = `${target.scale}`;
				break;
		}
	}
});

// init workspace
for(let key in workspaceConf) {
	workspaceConf[key] = workspaceConf[key];
}

// move the workspace while mouse dragging
qry("#workspaceContainer").addEventListener("mousedown", e => {
	const move = e => { // mousemove event
		workspaceConf.showingX -= e.movementX;
		workspaceConf.showingY -= e.movementY;
	}

	document.body.addEventListener("mousemove", move);
	document.body.addEventListener(
		"mouseup",
		e => document.body.removeEventListener("mousemove", move),
		{once: true}
	);
})

function createBlock({
	header,
	content,
	position: [x = workspaceConf.showingX, y = workspaceConf.showingY] = []
}){
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
			e => document.body.removeEventListener("mousemove", move),
			{once: true}
		);
	});

	// this is used to prevent unexpected workspace move
	block.addEventListener("mousedown", e => e.stopPropagation());



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
qry("#workspace").appendChild(createBlock({header: "Input"}));
