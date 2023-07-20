const qry = (...query) => document.querySelector(...query);
const qrys = (...query) => document.querySelectorAll(...query);

const workspace = qry("#workspace");

//block types
let blockTypes = {
	"Input": { // 這裡放 default value
		"parent": [],
		"child": [1, 2],
		"args": {
			"inputSize": [32, 2],				
			"outputSize": []
		}
	},	
	"Linear": { // 這裡放 default value
		"parent": [14],
		"child": [16],
		"args": {
			"inputSize": [],
			"outputSize": [32, 4096],
			"bias": 1
			}
	},
	"ReLu": {  // 這裡放 default value
		"parent": [15],
		"child": [17],
		"args": {
			"inputSize": [],
			"outputSize": [],
		}
	},
	"leakyReLU": {
		"parent": [15],
		"child": [17],
		"args": {
			"inputSize": [],
			"outputSize": [],
		}
	},
	"Sigmoid": {
		"parent": [15],
		"child": [17],
		"args": {
			"inputSize": [],
			"outputSize": [],
		}
	},
	"Tanh": {
		"parent": [15],
		"child": [17],
		"args": {
			"inputSize": [],
			"outputSize": [],
		}
	},
	"Concat":{
		"parent": [1, 2],
		"child": [4],
		"args": {
			"inputSize": [],
			"outputSize": [],
			"dim": 1
		}
	}
}



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

            block.dispatchEvent(new CustomEvent("redrawLines"));
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
	let header = document.getElementById("blockType").value, content = "";
    switch (header) {
        case "Input":
            content = "outputSize:";
            break;
		case "Linear":
            content = "outputSize:, bias:";
            break;
		case "ReLU":
            content = "mode:";
            break;
		case "leakyReLU":
            content = "mode:";
            break;
		case "Sigmoid":
            content = "mode:";
            break;
		case "Tanh":
            content = "mode:";
            break;
		case "Concat":
            content = "outputSize:, dim:";
            break;	
        default:
            content = " ";
            break;
    }
    qry("#workspace").appendChild(createBlock({ header, content }));
});

// create a Input block
qry("#workspace").appendChild(createBlock({ header: "Input" }));

// add lines while button#addLine is clicked
{
    const addLineBtn = qry("#addLine"),
        wsDiv = qry("#workspace");

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
		const redrawTheLine = (() => {
			const [port1, port2] = [document.createElement("div"), document.createElement("div")];
			[port1, port2].forEach(p => p.classList.add("port"));
				// ref: ./index.css, block .port
				// each port is a circle with radius equal to 5px.
			block1.querySelector(".outputPorts").appendChild(port1);
			block2.querySelector(".inputPorts").appendChild(port2);

			let svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
				// don't use doc~.createElement("svg" or "svg:svg") cuz that only
				// creates an HTML element with that tag name. The result will be 
				// an instance of HTMLUnknownElement, not a SVGElement.

            qry("#workspace").appendChild(svg);

            const
                abs = value => value > 0 ? value : -value,
                min = (...values) => Math.min(...values),
                max = (...values) => Math.max(...values);

            return () => {
                console.log("Drawing the line");
                const [c1, c2] = [port1, port2].map(port => {
                    // c for the center of a port, 
                    // relative to the workspace center (anchor)

					const anchor = wsDiv.getBoundingClientRect();
					const box = port.getBoundingClientRect();
					const
						x = (box.x + box.width / 2 - anchor.x) / workspaceConf.scale,
						y = (box.y + box.height / 2 - anchor.y) / workspaceConf.scale;
					return {x, y};
				});

                const area = {
                    up: min(c1.y, c2.y), // prevent using top as it's a keyword
                    down: max(c1.y, c2.y),
                    left: min(c1.x, c2.x),
                    right: max(c1.x, c2.x),
                    padding: 5
                };

                Object.assign(area, {
                    c1: { x: c1.x - area.left, y: c1.y - area.up },
                    c2: { x: c2.x - area.left, y: c2.y - area.up },

					// o means all or outer
					oUp: area.up - area.padding,
					oDown: area.down + area.padding,
					oLeft: area.left - area.padding,
					oRight: area.right + area.padding
						// things will be o right~
				});

                Object.assign(area, {
                    oC1: { x: c1.x - area.oLeft, y: c1.y - area.oUp },
                    oC2: { x: c2.x - area.oLeft, y: c2.y - area.oUp }
                });

                Object.entries({
                    "class": `lineContainer`,
                    width: `${area.oRight - area.oLeft}`,
                    height: `${area.oDown - area.oUp}`,
                    style: `
						position: absolute;
						left: ${area.left - area.padding}px;
						top: ${area.up - area.padding}px;
						pointer-events: none;
					`
                }).forEach(([attr, value]) => svg.setAttribute(attr, value));

				const setInnerXML = function(xml) {
					// ref: https://stackoverflow.com/a/9724151/10596093
					let tmpDoc = new DOMParser().parseFromString(xml, "application/xml");
					this.appendChild(
						this.outerDocument.adaptNode(tmpDoc.documentElement, true)
					);
				};

                svg.innerHTML = `
					<line 
						x1="${area.oC1.x}"
						y1="${area.oC1.y}"
						x2="${area.oC2.x}"
						y2="${area.oC2.y}"
					/>
				`;
            }
        })();

		[block1, block2].forEach(block => {
			block.addEventListener("redrawLines", redrawTheLine);
			block.dispatchEvent(new CustomEvent("redrawLines"));
				// after the ports added, other ports are sure to move,
				// so the other lines need to be redrawn as this line does.
		});

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