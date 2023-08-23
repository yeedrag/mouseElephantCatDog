/**
 * Notes for the codes:
 * - [Proxy]( https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Proxy )
 *   is used to intercept all modifications to specific objects. Therefore we can dispatch 
 *   events as the properties changed. 
 *
 * - [`Object.defineProperty()`]( https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Object/defineProperty )
 *   is used to precisely define a property of an object. They're default to read-only and
 *   not enumerable.
 *
 * - [Getter]( https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Functions/get )
 *   and [Setter]( https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Functions#getter_and_setter_functions )
 *   can be used to restrict or intercept an access to a property, too. Sometimes, it's
 *   much easier to used than Proxy and `Object.defineProperty()`.
 *
 *   In this code, we use getter-only properties to expose private members in classes as 
 *   read-only properties.
 *
 * Notes for the comments:
 * - Most (not all) of the comments starting with /** and ends with *\/ can be parsed as 
 *   jsDoc.
 *   - There seem to be some errors while jsDoc is parsing ES6 classes. Thus, we must
 *     explicitly specify `@memberof NAME_OF_THE_CLASS` for the properties/methods
 *     /members.
 */

/** 
 * A integer.
 * @typedef {number} integer
*/

/**
 * A utility function to call console.log much easier.
 * @param {...*} x - Just use this function as `console.log()`.
 */
const dbg = (...x) => console.log(...x);

/**
 * A utility function to deep copy things (especially for functions)
 * @func clone
 * @param {object} obj - The object to copy.
 * @return {object} The copied object.
 */
const clone = obj => {
	if(obj instanceof Array){
		return obj.map(ele => clone(ele))
	}else if(obj instanceof Object){
		return Object.fromEntries(Object.entries(obj).map(([key, value]) => [key, clone(value)]))
	}else if(obj instanceof String){
		return obj.toString()
	}else if(['string', 'number', 'function'].includes(typeof obj)){
		return obj // unlike an object, things in a function won't change, so we just reuse it
	}
};

/**
 * (Dummy) The type checker in type definition
 * @callback TypeChecker
 * @param {Block} inputBlock  - The block whose data will be transmitted to the `outputBlock`
 * @param {Block} outputBlock - The block who will get the data from input block.
 */

/**
 * (Dummy) A type for a block.
 *
 * TODO: Discuss for whether to create a class for it so that it's no longer an dummy concept.
 * TODO: How should the types of arguments be defined.
 *
 * @typedef {object} TypeSpec             
 * @prop {string}   name                  - The type's name.
 * @prop {string}   readableName          - A human readable name.
 * @prop {string}   description           - The description for the block; human readable
 *
 * @prop {object}   input                 - Information for the input block to be linked.
 * @prop {string}   input.type            - The type's name that the input block should meets.
 * @prop {string}   input.description     - Other requirements; human readable.
 * @prop {TypeChecker} input.checker      - A function to check whether the input block meets
 *   requirements besides the type. It should be invoked when users trying to connect a Line.
 *
 * @prop {object}   output                - Information for the output block to be linked.
 * @prop {string}   output.type           - The type's name that the output block should meets.
 * @prop {string}   output.description    - Other requirements; human readable.
 */

/** 
 * A class to contain types for blocks.
 */
class TypeManager {
	#registeredTypes = {};

	/** 
	 * Create the manager.
	 * @param {TypeSpec[]} [typeDefs] - A list of specifications of the types to be
	 * registered.
	 */
	constructor(specs = []) {
		specs.forEach(spec => 
			Object.defineProperty( // to define a type as constant (immutable)
				this.#registeredTypes,
				spec.name,
				{ value: clone(spec), enumerable: true } // clone to prevent further changes from outside
			)
		)
	}

	/**
	 * Register a type.
	 * @param {TypeSpec} spec - The specification of the type being registered.
	 * @throws {TypeError} When trying to register a type with the same name again.
	 *   More Info: [About the Error]( https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Errors/Cant_redefine_property )
	 */
	addType(spec) {
		// TODO: validate the `spec`
		Object.defineProperty(this.#registeredTypes, name, { value: clone(spec), enumerable: true });
	}

	/** 
	 * Get a type's spec by its name.
	 * @param {string} name - The name of the type who we are to get.
	 * @return {TypeSpec}
	 */
	getType(name) {
		return clone(this.#registeredTypes[name])
	}

	/** 
	 * Get an object of registered types.
	 * @return {Object} For each entries, the key is the type name {string}, 
	 *   and the value is the type definition {TypeSpec}
	 */
	listTypes() {
		return clone(this.#registeredTypes)
	}
}

/** 
 * An object that has adequate information for rebuilding a block.
 * This object can be parsed from a string with `JSON.parse()`, so can it
 * be stored as a string with `JSON.stringify()`.
 *
 * TODO: We may need to write functions for a more detailed conversion.
 *
 * @typedef {object} BlockExpr
 * @prop {string} type - The name of the block's type.
 *
 * TODO: Finish the definition here...
 *   - Input related
 *   - Output related
 */

/** 
 * A class that represents a block. It use the EventTarget feature, so 
 * we can use `.addEventListener()` and `.dispatchEvent()`.
 *
 * The concept of a Block is a visualized dummy operation. The operation reads
 * the input, behaves according to the arguments, and generate output in the
 * end. The input and output have specific formats and directions, and each 
 * arguments has their own type and value. 
 *
 * We stores the directions and values here, and summarize the formats and 
 * types with the Block's type. The detailed definition of the blocks are up 
 * there in TypeManager and TypeSpec.
 */
class Block extends EventTarget {
	/**
	 * Create the Block.
	 *
	 * @param {object}       options
	 * @param {TypeManager}  options.typeManager   - The type manager where the 
	 *   `options.source.type` is defined inside
	 * @param {BlockExpr}    [options.source]      - Information for (re)building
	 *   a block. Some (TODO: what exactly?) may be ignore here, such as 
	 *   input/output related information.
	 */
	constructor({ typeManager, source }){
		super();

		// initialize, so things won't panic if `source` is not a complete {BlockExpr}.
		source = { args: {}, ...source }

		// .type
		Object.defineProperty(this, "type", { value: source.type, enumerable: true });
			// by default: writable: false and enumerable: false
	
		// TODO: ... configure the block with other information from source
	}

	/**
	 * A magic object that stores the values of the block's arguments.
	 * For each entries, the key is attribute's name, and the value is its data.
	 * Once an entry is changed, the `argChange` event is dispatched on the Block.
	 *
	 * TODO: Discuss whether to create a class for arguments of an object.
	 *
	 * @fires Block#argChange - When any of the arguments' value is reassigned.
	 * @memberof Block.prototype
	 * @member args
	 * @const
	 * @type {Object}
	 */	
	#args = new Proxy({}, {
		set: (obj, key, value) => {
			/**
			 * An event that indicates that an argument's value is changed.
			 *
			 * @event Block#argChange
			 * @type {object}
			 * @property {object} detail
			 * @property {string} detail.argName - The name of the argument who is 
			 *   changed.
			 */
			this.dispatchEvent(
				new CustomEvent("argChange", {
					detail: { argName: key } 
				})
			);
			obj[key] = value;
			return true;
		}
	});
	get args(){ return this.#args; }


	/**
	 * This exports the block as an {BlockExpr} without the input/output
	 * related parts.
	 *
	 * @memberof Block.prototype
	 * @method exportObj
	 * @return {BlockExpr}
	 */
	exportObj(){
		return {
			type: clone(this.type),
			args: clone(this.args)

			// TODO: Are there something else that's suitable to be exported here?
			// Should we export the .pos?
		};
	}

	/**
	 * The visualized elements of a block are created alongside the block 
	 * itself.
	 *
	 * @memberof Block.prototype
	 * @member element
	 * @const
	 * @type {HTMLDivElement}
	 */
	#element = (()=>{
		const ele = document.createElement("div");
		ele.classList.add("block");
		ele.tabIndex = 0; // focusable

		// remove the whole block when the user press on delete
		ele.addEventListener("keydown", e => {
			if(e.target != e.currentTarget) return;
			if(e.key == "Delete")
				this.remove();
		})

		ele.innerHTML = `
			<div class="inputPorts">
				<div class="port addPort"></div>
			</div>
			<div class="header">${"NOT IMPLEMENTED HERE"}</div>
			<div class="content">${"NOT IMPLEMENTED HERE"}</div>
			<div class="outputPorts">
				<div class="port addPort"></div>
			</div>
		`;
		
		const [inputPortsEl, headerEl, contentEl, outputPortsEl] = ele.children;
		const [inputPortAddEl] = inputPortsEl.children;
		const [outputPortAddEl] = outputPortsEl.children;

		// move by dragging
		headerEl.addEventListener("mousedown", e => {
			const move = e => {
				this.pos.x += e.movementX / (this.manager.wsCfg.scale || 1);
				this.pos.y += e.movementY / (this.manager.wsCfg.scale || 1);
			}
			document.body.addEventListener("mousemove", move);
			document.body.addEventListener("mouseup", e => {
				document.body.removeEventListener("mousemove", move);
			}, { once: true })
		});
	
		// TODO: Configure the two *AddPortEl, so they start a session after clicked. 

		return ele;
	})();
	get element(){ return this.#element }

	/**
	 * The components of the Block.prototype.element, including
	 *   - `inputPortsEl`
	 *       The place to hold the (Line.prototype.outputPortEl)s
	 *       from the Lines attached.
	 *   - `headerEl`
	 *       The field for the Block's title, which usually is the 
	 *       TypeSpec.readableName where the TypeSpec is looked up with the Block's 
	 *       type in the TypeManager that's used to create the Block.
	 *   - `contentEl`
	 *      The field to hold the elements of the Block's arguments.
	 *   - `outputPortsEl`
	 *       The place to hold the (Line.prototype.inputPortEl)s
	 *       from the Lines attached.
	 *   - `inputPortAddEl`
	 *       A dummy port in `inputPortsEl` that user can click
	 *       to start a line connection session. (TODO: See Block.prototype.#element)
	 *   - `outputPortAddEl`
	 *       A dummy port in `outputPortsEl` that user can click
	 *       to start a line connection session. (TODO: same as the above)
	 *
	 * TODO: To make the code more stable, maybe we should 
	 *   - prevent the *El.remove() methods from being called
	 *   - declare private member for each *El, and set them up in constructor().
	 *     Then, we can make our getters simply refer to them, without calling 
	 *     querySelector() whenever we need them.
	 *
	 * TODO: Create elements to visualize the arguments, and re-render them.
	 *
	 * @memberof Block.prototype
	 * @member *El
	 * @const
	 * @type {HTMLDivElement}
	 */
	get inputPortsEl(){ return this.element.querySelector(":scope > .inputPorts") }
	get headerEl(){ return this.element.querySelector(":scope > .header") }
	get contentEl(){ return this.element.querySelector(":scope > .content") }
	get outputPortsEl(){ return this.element.querySelector(":scope > .outputPorts") }

	get inputPortAddEl(){ return this.inputPortEl.querySelector(":scope > .addPort") }
	get outputPortAddEl(){ return this.outputPortEl.querySelector(":scope > .addPort") }

	/* -- START OF for BlockManager -- */
	/**
	 * This method is for {BlockManager} and shouldn't be called directly.
	 * See BlockManager.prototype.addBlock().
	 *
	 * @memberof Block.prototype
	 * @method _assignManager
	 * @param {object}       options
	 * @param {BlockManager} options.manager  - The manager to be assigned to this Block.
	 * @param {integer}      options.id       - The id that this Block has in the manager.
	 */
	_assignManager({manager, id}){
		if(this.#manager != undefined)
			throw "Block already has a manager.";

		[this.#manager, this.#id] = [manager, id];

		// TODO: ... to be continued
	}

	/**
	 * This method is for {BlockManager} and shouldn't be called directly.
	 * See BlockManager.prototype.removeBlockById().
	 *
	 * @memberof Block.prototype
	 * @method _unassignManager
	 */
	_unassignManager(){
		if(this.#manager == undefined)
			throw "Block doesn't have a manager.";

		[...this.#inputLines, ...this.#outputLines].forEach(line => line.remove());
		this.#manager = this.#id = undefined;
	}

	/**
	 * The manager of the Block.
	 *
	 * @type {undefined|BlockManager}
	 * @memberof Block.prototype
	 * @member manager
	 * @const
	 * @type {BlockManager}
	 */
	#manager = undefined;
	get manager(){ return this.#manager }

	/**
	 * A shorthand to call removeBlockById() on the BlockManager of this Block.
	 * This function removes the Block completely, but the Block should be 
	 * reusable. 
	 *
	 * @methodof Block.prototype
	 * @method remove
	 */
	remove(){ this.#manager?.removeBlockById(this.id) }

	/**
	 * The id that this Block has in its BlockManager. It is actually the 
	 * index of this Block in the BlockManager's blockList array.
	 * See `BlockManager.prototype.addBlock()`.
	 *
	 * @memberof Block.prototype
	 * @member id
	 * @const
	 * @type {integer}
	 */
	#id = null;
	get id(){ return this.#id }

	/**
	 * The position of the Block's element in the workspace.
	 * It's a magic member that provides easy assignment and retrievement.
	 *
	 * Use `pos.x` or `pos[0]` to assign/fetch the horizontal position; `pos.y` 
	 * and `pos[1]` are for vertical position. You can also do 
	 * `pos = { x: 12, y: 24 }` to set it quickly. `pos = [12, 24]` has the 
	 * same effect.
	 *
	 * @memberof Block.prototype
	 * @member pos
	 * @type {object}
	 */
	#pos = new Proxy({ x: 0, y: 0 }, {
		set: (target, key, value) => {
			if(!["0", "1", "x", "y"].includes(key)) 
				return false;

			if(key == "0" || key == "x")
				target.x = value;
			if(key == "1" || key == "y")
				target.y = value;

			this.element.style["translate"] = `${target.x}px ${target.y}px`;

			/**
			 * This event fires when the position changed.
			 * @event Block#posChange
			 * @type {undefined}
			 */
			this.dispatchEvent(new CustomEvent("posChange"));
			return true;
		},
		get: (target, key) => {
			if(key == "x" || key == "0") return target.x;
			if(key == "y" || key == "1") return target.y;
		}
	}); // initialized in constructor()
	set pos(value){
		let has = x => Object.keys(value).includes(x);
		if(has("x") || has("0")) this.#pos.x = value.x ?? value[0];
		if(has("y") || has("1")) this.#pos.y = value.y ?? value[1];

		if(!has("x") && !has("y") && !has("0") && !has("1"))
			return false;
		return true;
	}
	get pos(){ return this.#pos }
	
	/* -- START OF for Line -- */
	// this array stores the lines whose outputPorts are treated as this Block's inputPorts.
	#inputLines = [];

	/**
	 * This method is for Line and shouldn't be called directly. See 
	 * `Line.prototype._setOutputBlock()`.
	 *
	 * This method is used to add a line at the input side of this Block.
	 *
	 * @memberof Block.prototype
	 * @method _assignInputLine
	 * @param {Line}    line     - The Line whose output port is to be inserted into this Block's
	 *   inputPortsEl.
	 * @param {integer} [index]  - The position that the port should be inserted.
	 */
	_assignInputLine(line, index=this.#inputLines.length) {
		// TODO: Discuss if we should remove all the Els and 
		this.inputPortsEl.insertBefore(
			line.outputPort,
			this.inputPortsEl.children[index] || this.inputPortAddEl || null
				// insertBefore(): when reference node is null, insert at the end
		);
		this.#inputLines.splice(index, 0, line);

		// TODO: Fires `inputPortsChange` event
	}

	/**
	 * This method removes a Line and its port from the Block's input part.
	 *
	 * @param {Line} line - The Line to be removed.
	 */
	_unassignInputLine(line){
		// TODO: Discuss what to do if the line isn't in the Block's input part?
		//   - Option 1: Just return.
		//   - Option 2: Throw custom error.
		//   - Option 3: Let the error be (like now).

		let index = this.#inputLines.indexOf(line);
		this.#inputLines.splice(index, 1);

		// TODO: Discuss whether we should check if the port is in our inputPortsEl
		//   and what to do if it's not.
		line.outputPort.remove();

		// TODO: Fires `inputPortsChange` event
	}

	// TODO: Document these as the input part of Block.
	// TODO: Do the TODO s written in the input part here.
	#outputLines = [];
	// ._assignOnputLine(), for Line
	_assignOnputLine(line, index=this.#outputLines.length) {
		this.outputPortsEl.insertBefore(
			line.inputPort,
			this.outputPortsEl.children[index] || this.outputPortAddEl || null
				// insertBefore(): when reference node is null, insert at the end
		);
		this.#outputLines.splice(index, 0, line);
	}

	// ._unassignOnputLine(), for Line
	_unassignOnputLine(line){
		let index = this.#outputLines.indexOf(line);
		this.#outputLines.splice(index, 1);
		line.inputPort.remove();
	}

	// ._exportObjForManager(), for BlockManager
	// TODO: Write the function that exports a BlockExpr but with input, output,
	//   and pos parts included. This function should be called by BlockManager 
	//   when exporting things.
	/* -- END OF for Line -- */
	/* -- END OF for Block Manager -- */
}
/**
 * A class that represents a connection between two blocks.
 */
class Line {
	/**
	 * This constructor is for BlockManager and shouldn't be called directly.
	 * See `BlockManager.prototype.genLine()`.
	 *
	 * This method creates a line.
	 *
	 * TODO: Consider to add other parameters so that it'll be easier to import/export.
	 *
	 * @param {BlockManager} blockManager - The block manager that creates this Line.
	 */
	constructor(blockManager){
		if(!blockManager instanceof BlockManger)
			throw "param blockManager should be an instance of BlockManager";

		this.#manager = blockManager;
		this.#manager.ws.appendChild(this.element);

		// the inputPort to the line is the outputPort to the input block;
		// vice versa.
		["inputPortEl", "outputPortEl"].forEach(port => {
			Object.defineProperty(this, port, { value: document.createElement("div") });
			this[port].classList.add("port");
			this[port].addEventListener("pointerdown", async e => {
				const ptrDownTime = new Date();
				await new Promise(res => this[port].addEventListener("pointerup", res, {once: true}));
				const ptrUpTime = new Date();

				if(ptrUpTime - ptrDownTime > 200)
					return; // not a 'click'
				this.remove();
			})
		});
	}

	/**
	 * The manager of the Line - i.e., the manager who creates this Line and displays this line in
	 * their workspace.
	 *
	 * @memberof Line.prototype
	 * @member manager
	 * @const
	 * @type {undefined|BlockManager}
	 */
	#manager = undefined;
	get manager(){ return this.#manager }

	/**
	 * This function removes the Line completely. The line isn't reusable later.
	 *
	 * @memberof Line.prototype
	 * @method remove
	 */
	remove() {
		this._setInputBlock();
		this._setOutputBlock();
		this.manager._removeLine(this);
		this.element.remove();
	}

	/**
	 * The id of the input block.
	 *
	 * @memberof Line.prototype
	 * @member inputBlockId
	 * @type {integer|undefined}
	 * @const
	 */
	#inputBlockId = undefined;
	get inputBlockId() { return this.#inputBlockId }

	/**
	 * The input block.
	 *
	 * @memberof Line.prototype
	 * @member inputBlock
	 * @type {Block|undefined}
	 * @const
	 */
	get inputBlock() { return this.#manager.getBlocks()[this.#inputBlockId] }

	/**
	 * This method is for Block and Line.remove(), and shouldn't be called directly. 
	 * It should be called by the one who started Line connection session.
	 *
	 * This method insert the Line's inputPort into a Block's outputLines array with
	 * a give index. It automatically unset any probably existing old inputBlock.
	 *
	 * See: `Line.prototype.remove()`.
	 * See: `Block.prototype.#element`. (TODO: This method is planned to be used in
	 * the `*AddPortEl` parts for line connection sessions.)
	 *
	 * @memberof Line.prototype
	 * @method _setInputBlock
	 * @param blockId - The id of the input block. Leave it empty means to unset
	 * the inputBlock only.
	 * @param [index] - The index in `this.inputBlock.outputLines` array that this Line 
	 * should be inserted into.
	 * @return {bool} Whether it's successfully set. True if successful, and false
	 * if failed. (TODO: not implemented.)
	 */
	_setInputBlock(blockId, index) {
		// remove the old block
		if(this.#inputBlockId)
			this.inputBlock._unassignOutputLine(this);

		// TODO: If this.outputBlock exists, run the type checker for the inputBlock
		// to see if this will be a valid line. If valid, keep going anf returns 
		// true later; if not, returns just return false.

		// set the id and configure the line
		this.#inputBlockId = blockId;
		if(blockId)
			this.inputBlock._assignOutputLine(this, index);

		// TODO: Listen to events of this.inputBlock that indicates the possibility of 
		// layout change of it, and then call this.redraw() on events. Of course, 
		// remove the listener while removing the old block.	
	}

	/** 
	 * The id of the output Block.
	 *
	 * @memberof Line.prototype
	 * @member outputBlockId
	 * @const
	 * @type {undefined|integer}
	 */
	#outputBlockId = undefined;
	get outputBlockId() { return this.#outputBlockId }

	/**
	 * The output block.
	 *
	 * @memberof Line.prototype
	 * @member outputBlock
	 * @const
	 * @type {Block|undefined}
	 */
	get outputBlock() { return this.#manager.getBlocks()[this.#outputBlockId] }

	/**
	 * See Line.prototype._setInputBlock().
	 *
	 * @memberof Line.prototype
	 * @method setOutputBlock
	 */
	_setOutputBlock(blockId, index) {
		// remove the old block
		if(this.#outputBlockId)
			this.outputBlock._unassignInputLine(this);

		// set the id and configure the line
		this.#outputBlockId = blockId;
		if(blockId)
			this.outputBlock._assignInputLine(this, index);

		// TODO: See TODO: in Line.prototype._setInputBlock()
	}

	/**
	 * The SVG Element visualizing the Line. It's automatically added to the 
	 * workspace. See {BlockManager.prototype.genLine()}.
	 *
	 * @memberof Line.prototype
	 * @member element
	 * @const
	 * @type {SVGSVGElement}
	 */
	#element = (()=>{
		const ele = document.createElementNS("http://www.w3.org/2000/svg", "svg");
			// don't use doc~.createElement("svg" or "svg:svg") cuz that only
			// creates an HTML element with that tag name. The result will be 
			// an instance of HTMLUnknownElement, not a SVGElement.
		ele.classList.add("lineContainer");
		return ele;
	})();
	get element(){ return this.#element }

	/**
	 * This function redraws the Line in the workspace.
	 *
	 * @memberof Line.prototype
	 * @method redraw
	 */
	redraw() {
		this.element.innerHTML = "";
		if(!this.inputPort.checkVisibility() || !this.outputPort.checkVisibility())
			// TODO: rewrite this line cuz `checkVisibility` is only in a draft, not yet a living standard.
			//   We may refer to:
			//     - [The Draft]( https://drafts.csswg.org/cssom-view/#dom-element-checkvisibility )
			//     - Element.prototype.getBoundingClientRect()
			//     - Element.prototype.getClientRects()
			return;

		// some useful utilities
		const abs = x => x > 0 ? x : -x;
		const min = (...v) => Math.min(...v);
		const max = (...v) => Math.max(...v);

		// get the centre points of ports
		const [c1, c2] = [this.inputPort, this.outputPort].map(port => {
			const wsPosition = this.#manager.getBoundingClientRect();
			const portBox = port.getBoundingClientRect();

			const x = 
				(portBox.x + portBox.width / 2 - anchor.x)
				/ this.#manager.wsCfg.scale;
			const y =
				(portBox.y + portBox.height / 2 - anchor.x)
				/ this.#manager.wsCfg.scale;
			return {x, y}
		});

		// # start to set for this.element
		// here is the ideal area of this.element if line's width is 0
		const area = {
			up: min(c1.y, c2.y),
			down: max(c1.y, c2.y),
			left: min(c1.x, c2.x),
			right: max(c1.x, c2.x),
			padding: 5
				// line has it's width, so leave some space
				// ref: ./index.css, .lineContainer
		}

		// here is the real area of this.element, where 'o' means all or outer
		Object.assign(area, {
			oUp: area.up - area.padding,
			oDown: area.down + area.padding,
			oLeft: area.left - area.padding,
			oRight: area.right + area.padding
				// things will be o right~
		});

		// here is the positions of the line's two ends relative to the real
		// area, and the width and height of the real area
		Object.assign(area, {
			oC1: { x: c1.x - area.oLeft, y: c1.y - area.oUp },
			oC2: { x: c2.x - area.oLeft, y: c2.y - area.oUp },

			oWidth: oRight - oLeft,
			oHeight: oDown - oUp
		});

		// configure this.element
		Object.entries({
			"class": "lineContainer",
			"width": `${oWidth}`,
			"height": `${oHeight}`,
			"style": `
				position: absolute;
				translate: ${oLeft}px ${oTop}px;
			`
		}).forEach(([attrName, attrValue]) => {
			this.element.setAttribute(attrName, attrValue)
		});
		this.element.innerHTML =
			`<line
				x1="${area.oC1.x}"
				y1="${area.oC1.y}"
				x2="${area.oC2.x}"
				y2="${area.oC2.y}"
			/>`;
	}
}

/** 
 * An object that has adequate information for rebuilding a workspace.
 * This object can be parsed from a string with `JSON.parse()`, so can it
 * be stored as a string with `JSON.stringify()`.
 *
 * TODO: We may need to write functions for a more detailed conversion.
 *
 * @typedef {object} WsExpr
 * @prop {blockExpr[]} blocks - A list of the blocks it contains.
 *
 * TODO: Finish the definition here...
 */


/**
 * A class that holds a list of Blocks and Lines, and controls the elements that
 * visualize them.
 */
class BlockManager extends EventTarget {
	/**
	 * Creates a BlockManager.
	 *
	 * @param options
	 * @param {HTMLElement} options.workspaceContainer - An element to wrap the 
	 *   workspace. This element may
	 *   - be empty, so the function will insert a <div\> into it as the workspace.
	 *   - have at least a <div\> or <main\>.
	 *     - If some of these <div\> or <main\> has the "workspace" class, the 
	 *       first of them (with "workspace") will be chosen as the workspace 
	 *       element.
	 *     - If none of these has the "workspace" class, then the first of the 
	 *       <div\> or <main\>s will be chosen as the workspace elements.
	 * @param {WsExpr} [options.source] - This is for quick import/export.
	 */ 
	constructor({ workspaceContainer, source = {} }) {
		super();

		// .#wsC 
		if(!(workspaceContainer instanceof HTMLElement))
			throw "please provide an element as workspaceContainer";
		this.#wsC = workspaceContainer;

		// .#ws
		let wsCandidate = [...this.#wsC.querySelectorAll(":scope > :is(div, main)")]
			.filter(ele => ele.children.length == 0)
			.sort((e1, e2) => e1.classList.has("workspace") ? -1 : 1)[0];
		if(wsCandidate){
			this.#ws = wsCandidate;
		}else{
			this.#ws = document.createElement("div");
			this.#wsC.appendChild(this.#ws);
		}

		// .#wsCfg
		Object.entries(this.#wsCfg).forEach(([key, value]) => {
			this.#wsCfg[key] = value;
		})

		/* to import from `source` */	
		// TODO: import from `source`

		// now apply .wsCfg onto .ws and .wsC
		Object.keys(this.wsCfg).forEach(key => this.wsCfg[key] = this.wsCfg[key]);

		// allow users to scale the workspace by scrolling
		// and move it by dragging
		this.#initWorkspace({ ws: this.ws, wsC: this.wsC, wsCfg: this.wsCfg });
	} // constructor()

	// this function is part of the constructor(). We moved it out for clarity.
	#initWorkspace({ ws, wsC, wsCfg }) {
		// move the workspace while mouse dragging
		wsC.addEventListener("mousedown", e => {
			if (e.target != wsC) return;
			// prevent unexpected workspace movement (e.g. while blocks being dragged)

			const move = e => { // mousemove event
				wsCfg.movementX += e.movementX;
				wsCfg.movementY += e.movementY;
			}

			document.body.addEventListener("mousemove", move);
			document.body.addEventListener(
				"mouseup",
				e => document.body.removeEventListener("mousemove", move),
				{ once: true }
			);
		});

		// scale the workspace while scrolling
		{
			let mousePosInWsC = [0, 0]; // mouse's position in workspace container
			wsC.addEventListener("mousemove", e => {
				let wsCBox = wsC.getBoundingClientRect();
				mousePosInWsC = [e.x - wsCBox.x, e.y - wsCBox.y];
					// we can't just set that to [e.layerX, e.layerY] 'cuz
					// the "layer" may be any block the cursor's hovering on
			});

			wsC.addEventListener("wheel", e => {
				let oldScale = wsCfg.scale;

				wsCfg.scale *= 3 ** (e.deltaY * -0.001);
				let newScale = wsCfg.scale = Math.max(0.05, wsCfg.scale);

				// use mouse's pos as the scaling center
				let vec = [
					wsCfg.movementX - mousePosInWsC[0],
					wsCfg.movementY - mousePosInWsC[1],
				];

				let vecScale = newScale / oldScale;
				wsCfg.movementX = vec[0] * vecScale + mousePosInWsC[0];
				wsCfg.movementY = vec[1] * vecScale + mousePosInWsC[1];
			}, { passive: true });
		}
	}

	/**
	 * The workspace, where Blocks and Lines are displayed inside.
	 *
	 * @memberof BlockManager.prototype
	 * @member ws
	 * @const
	 * @type {HTMLElement}
	 */
	#ws = undefined; // init in constructor
	get ws() { return this.#ws }

	/**
	 * The container of the workspace.
	 *
	 * @memberof BlockManager.prototype
	 * @member wsC
	 * @const
	 * @type {HTMLElement}
	 */
	#wsC = undefined; // init in constructor
	get wsC() { return this.#wsC }

	/**
	 * The configuration of the workspace. It's a special object that configure 
	 * itself and `this.ws` automatically.
	 *
	 * @memberof BlockManager.prototype
	 * @member wsCfg
	 * @const
	 * @type {Object}
	 * @property {number} movementX - The horizontal position of the ws relative to 
	 *   the wsC.
	 * @property {number} movementY - The vertical position of the ws relative to 
	 *   the wsC.
	 * @property {number} scale     - How much is the ws zoomed.
	 * @property {number} showingX  - The horizontal position of the top-left point
	 *   in ws. i.e., if I want to put a Block's element in the ws, and align the 
	 *   top-left point of the Block with the top-left point of the wsC, what should
	 *   the Block's `pos.x` be?
	 * @property {number} showingY  - Similar to the above, but it's the vertical
	 *   position and the Block's `pos.y`.
	 */
	#wsCfg = new Proxy(
		{ movementX: 0, movementY: 0, scale: 1},
		{ set: (target, key, value) => {
			target[key] = value;

			if (key == "scale") {
				// apply scale
				this.ws.style["scale"] = `${target.scale}`;
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

			if (["movementX", "movementY", "showingX", "showingY"]) {
				// i.e., if movement changed

				this.ws.style["translate"] = `${target.movementX}px ${target.movementY}px`;
					// note: the "translate"d amounts seem not to be affected by "scale"
			}

			return true; // set handler should return true if success.
		} }); // new Proxy()
		// init in constructor
	get wsCfg(){ return this.#wsCfg }

	/**
	 * An array of the Blocks that this BlockManager contains. The `id` of a Block is 
	 * actually the index of it in this array, so this may not be a continuous array
	 * (i.e., some of the elements may be empty).
	 *
	 * @memberof BlockManager.prototype
	 * @member Block
	 * @const
	 * @type {Block[]}
	 */
	#blocks = [];
	get blocks() { return [...this.#blocks] }

	/**
	 * Add a Block to this BlockManager. This automatically assigns an id to the 
	 * Block, which is actually the index of the Block in the `blocks` Array.
	 *
	 * @memberof BlockManager.prototype
	 * @method addBlock
	 * @param {Block} block - The Block to be added.
	 */
	addBlock(block) {
		if(!block instanceof Block)
			throw "addBlock(block) can only accept a Block as the block argument.";

		// determine id by subtract 1 from the return value of 
		// Array.prototype.push(), which is the new length
		const id = this.#blocks.push(block) - 1;

		// configure the Block
		block._assignManager({manager: this, id});
		this.#ws.appendChild(block.element);
		block.pos = [this.wsCfg.showingX, this.wsCfg.showingY];
	}

	/**
	 * Remove a block from this BlockManager with a certain id.
	 *
	 * @memberof BlockManager.prototype
	 * @method removeBlockById
	 * @param {integer} id - The `id` of the Block to be removed..
	 */
	removeBlockById(id){
		if(!['number', 'string'].includes(typeof id))
			throw "please specify the id";

		this.#blocks[id].element.remove();
		this.#blocks[id]._unassignManager();
		delete this.#blocks[id];
	}

	/**
	 * An array of lines that's created by `this.genLine()`.
	 *
	 * @memberof BlockManger.prototype 
	 * @member lines
	 * @const 
	 * @type {Line[]}
	 */
	#lines = [];
	listLine(){ return [...this.#lines] }

	/**
	 * Create a Line() instance and add its element into `this.ws`.
	 *
	 * @memberof BlockManager.prototype
	 * @method genLine
	 * @return {Line}
	 */
	genLine(){
		let newLine = new Line(this);
			// Line() would add its element into this.ws
		this.#lines.push(newLine);

		// TODO: Add the newLine.element to the workspace.

		return newLine;
	}

	/**
	 * This method is for Line and shouldn't be called directly.
	 * See: `Line.prototype.remove()`
	 *
	 * Remove a Line from this BlockManager.
	 *
	 * @memberof BlockManager.prototype
	 * @method _removeLine
	 * @param {Line} line - The line to be removed.
	 */
	_removeLine(line) {
		this.#lines.splice(this.#lines.indexOf(line), 1);
	}
}

export { TypeManager, Block, Line, BlockManager }
