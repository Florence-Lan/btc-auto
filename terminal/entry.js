// Preserve bookmarks from the former combined terminal.
const destination = {"#btc": "./btc.html", "#stocks": "./stocks.html"}[location.hash];
if (destination) location.replace(destination);
