export default function(eleventyConfig) {
	// Copy CSS file to output
	eleventyConfig.addPassthroughCopy("styles.css");

	// Copy attachments folder to output
	eleventyConfig.addPassthroughCopy("blog/attachments");
	
	// Copy embeds for blog posts
	eleventyConfig.addPassthroughCopy("blog/embeds");

	// Set default layout for all markdown files
	eleventyConfig.addGlobalData("layout", "base.njk");

	// Add date filter for nice formatting
	eleventyConfig.addFilter("niceDate", (dateObj) => {
		return dateObj.toLocaleDateString('en-US', {
			year: 'numeric',
			month: 'long',
			day: 'numeric'
		});
	});

	// Add string helper filters
	eleventyConfig.addFilter("startsWith", (str, prefix) => {
		return str && str.startsWith(prefix);
	});

	// Create status collection
	eleventyConfig.addCollection("status", function(collectionApi) {
		return collectionApi.getFilteredByTag("status").sort((a, b) => {
			return b.date - a.date; // Sort by date descending
		});
	});

	// Create favorites collection
	eleventyConfig.addCollection("favorites", function(collectionApi) {
		return collectionApi.getAll().filter(item => {
			return item.url && item.url.startsWith('/favorites/') && item.url !== '/favorites/';
		}).sort((a, b) => {
			return b.date - a.date; // Sort by date descending
		});
	});

	// Create posts collection (sorted by date)
	eleventyConfig.addCollection("posts", function(collectionApi) {
		return collectionApi.getFilteredByTag("post").sort((a, b) => {
			return b.date - a.date; // Sort by date descending (newest first)
		});
	});

	// Transform relative image paths for posts
	eleventyConfig.addTransform("rewriteImagePaths", function(content, outputPath) {
		if (outputPath && outputPath.endsWith(".html") && outputPath.includes("/blog/")) {
			// Replace relative attachments/ paths with ../attachments/
			return content.replace(/src="attachments\//g, 'src="../attachments/');
		}
		return content;
	});

	return {
		dir: {
			input: ".",
			includes: "_includes",
			output: "_site"
		}
	};
}
