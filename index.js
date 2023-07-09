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

		return true; // set handler should return true if success.
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

// add lines while button#addLine is clicked
{
	const addLineBtn = qry("#addLine"), wsDiv = qry("#workspace");

	const prepareForAnotherProcess = () => {
		// "process" here and below means the period while connecting two blocks

		addLineBtn.innerHTML = `Connect Blocks`;
		addLineBtn.addEventListener("click", addLine, {once: true});
	};

	const addLine = async e => { // click event
		addLineBtn.innerHTML = `Stop Connecting Blocks`;

		// cancel the process if clicked again
		// all the long-term processes should listen to this AbortController.signal
		const addLineProcess = new AbortController();
		addLineBtn.addEventListener("click", e => {
			addLineProcess.abort();
			prepareForAnotherProcess();
		}, {once: true});

		// userSelectedABlock() is defined later
		let block1 = await userSelectedABlock({signal: addLineProcess.signal})
			.catch(err => {	
				if(err.name != "aborted")
					console.log("Error: ", err);
			});
		let block2 = await userSelectedABlock({signal: addLineProcess.signal})
			.catch(err => {
				if(err.name != "aborted")
					console.log("Error: ", err);
			});

		if(addLineProcess.signal.aborted) return prepareForAnotherProcess();
		if(block1 == block2){
			alert("You can't connect a block to itself!"),
			prepareForAnotherProcess();
			return;
		}

		// now draw a line for the two blocks
		console.log(block1, block2); // to do

		prepareForAnotherProcess();
	};

	prepareForAnotherProcess();

	const userSelectedABlock = ({signal: signal}) => {
		return new Promise((resolve, reject) => {
			if(signal.aborted) fail({name: "aborted"});
			signal.addEventListener("abort", e => reject({name: "aborted"}), {once: true});

			const findTheBlock = e => { // click event
				// finding block
				let tmp = e.target;
				while(!tmp.classList.contains("block")){
					if(tmp == wsDiv){
						listenToClick();
						alert("Please click on a block");
						return;
					}
					tmp = tmp.parentElement;
				}
				const block = tmp;
				resolve(block);
			};

			const listenToClick = () => wsDiv.addEventListener(
				"click",
				findTheBlock,
				{signal: signal, once: true}
			);

			listenToClick();
		});
	}
}
