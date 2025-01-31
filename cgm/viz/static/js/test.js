console.log("=== TEST SCRIPT LOADED ===");

// Test function
function testConsole() {
    console.log("=== TEST FUNCTION CALLED ===");
}

// Call test function immediately
testConsole();

// Call test function after a delay
setTimeout(testConsole, 1000);

// Add test function to window
window.testConsole = testConsole; 