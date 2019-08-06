<template>
  <v-app>
    <Logo />
    <v-content>
      <nuxt />
    </v-content>
    <v-footer
      app
    >
      <span>スパークジョイ &copy; 2019</span>
    </v-footer>
  </v-app>
</template>

<script>
import Logo from '@/components/Logo'

export default {
  data() {
    return {
      title: 'ShowTime'
    }
  },
  components: {
    Logo
  }
}
</script>

<style>
  body {
    background-color: #fafafa;
  }
</style>
