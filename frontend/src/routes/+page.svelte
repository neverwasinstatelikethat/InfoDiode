<script lang="ts">
	import LandingHero from '$lib/components/LandingHero.svelte';
	import LandingFeatures from '$lib/components/LandingFeatures.svelte';
</script>

<div class="min-h-screen">
	<LandingHero />
	<LandingFeatures />

	<!-- Footer -->
	<footer class="border-t border-border py-8 mt-8">
		<div class="max-w-7xl mx-auto px-4 lg:px-8">
			<div class="flex flex-col sm:flex-row items-center justify-between gap-4">
				<div class="flex items-center gap-2">
					<div class="h-5 w-5 rounded bg-accent/10 flex items-center justify-center">
						<svg class="w-3 h-3 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
							<path stroke-linecap="round" stroke-linejoin="round" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
						</svg>
					</div>
					<span class="text-sm text-text-secondary">InfoDiode</span>
				</div>
				<div class="text-xs text-text-tertiary">
					Нейросетевое извлечение данных из мнемосхем SCADA
				</div>
			</div>
		</div>
	</footer>
</div>
