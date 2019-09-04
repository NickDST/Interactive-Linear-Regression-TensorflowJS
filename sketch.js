let x_vals = [];
let y_vals = [];

let m, b;

const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);

function mousePressed(){
	let x = map(mouseX, 0, width, 0, 1);
	let y = map(mouseY, height, 0, 1, 0);

	x_vals.push(x);
	y_vals.push(y);
}

function setup() {
	createCanvas(800, 800);
	
	m = tf.variable(tf.scalar(random(1)));
	b = tf.variable(tf.scalar(random(1)));

}

function predict(x){
	//we need to convert this to a tensor
	const xs = tf.tensor1d(x);
	//formula of a line y = mx + b
	const ys = xs.mul(m).add(b);
	return ys
}


function loss(pred, labels){
	return pred.sub(labels).square().mean();
	// (pred, label) => pred.sub(label).square().mean();
}




function draw(){

	// function train(){
	// 	loss(predict(x_vals), ys);
	// }
tf.tidy(() => {
	if(x_vals.length > 0){

	const ys = tf.tensor1d(y_vals);

	// Correct way to do 
	optimizer.minimize(() => loss(predict(x_vals), ys));
	}
});
	background(50);

	// stroke(255);
	stroke(255, 255, 0);
	strokeWeight(10);
	for(let i = 0; i < x_vals.length; i++){
		let px = map(x_vals[i], 0, 1, 0, width);
		let py = map(y_vals[i], 1, 0, height, 0);
		point(px, py);
	}


	tf.tidy(() => {


	const xs = [0,1];
	const ys = predict(xs);
	// ys.print();

	let x1 = map(xs[0], 0, 1, 0, width);
	let x2 = map(xs[1], 0, 1, 0, width);


	let lineY = ys.dataSync();
	// console.log(lineY);
	let y1 = map(lineY[0], 0, 1, 0, height);
	let y2 = map(lineY[1], 0, 1, 0, height);
	strokeWeight(2);
	line(x1, y1, x2, y2);
});

// finding memory leaks
console.log(tf.memory().numTensors);

}