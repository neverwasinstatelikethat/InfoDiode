<script lang="ts">
	import '../app.css';
	import { page } from '$app/state';
	import AppShell from '$lib/components/AppShell.svelte';

	let { children } = $props();

	// Страницы авторизации рендерятся без AppShell
	let isAuthPage = $derived(
		page.url?.pathname?.startsWith('/auth') ?? false
	);
</script>

{#if isAuthPage}
	{@render children()}
{:else}
	<AppShell>
		{@render children()}
	</AppShell>
{/if}
