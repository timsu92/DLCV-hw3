import { chromium } from "playwright";
const browser = await chromium.launch();
const page = await browser.newPage();
const htmlPath = new URL("../src/index.html", import.meta.url).pathname;
await page.goto(`file://${htmlPath}`);
await page.pdf({ path: "report.pdf", format: "A4", printBackground: true });
await browser.close();
console.log("PDF generated: report.pdf");
