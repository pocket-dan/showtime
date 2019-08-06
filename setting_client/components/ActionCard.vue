<template>
  <v-card>
    <div class="card">
      <img :src="card.image" alt="">
    </div>
  </v-card>
</template>

<script>
  export default {
    props: {
      card: {
        type: Object,
        required: true
      }
    }
  }
</script>

<style lang="scss" scoped>
  .card {
    width: 100px;
    height: 100px;
    margin: 0 auto;
    img {
      width: 100%;
      height: 100%;
    }
  }
</style>
